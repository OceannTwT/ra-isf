import json
import logging
import re
import string

import os
import argparse
import csv
import json
import logging
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

from config import args
from contriever_config import c_args
from collections import Counter
from utils import write_json, print_now, load_data, print_exp, mkpath
from source.model.llama2_predict import predict, model_init

from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig

from retrieval_contriever.passage_retrieval import embed_queries, index_encoded_data, add_embeddings, validate, add_passages, add_hasanswer, load_data
from source.arch.passage_relevance.pr import Passage_Relevance_Model
from source.arch.self_knowledge.sk import Self_Knowledge_Model
from source.arch.task_decomposition.td import Task_Decomposition_Model


import retrieval_contriever.src.index
import retrieval_contriever.src.contriever

def load_dataset(data_path):
    dataset = list()
    with open(data_path, 'r', encoding='UTF-8') as f:
        for idx, line in enumerate(f):
            datas = json.loads(line)
            dataset.append(datas)
    return dataset

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s.strip()))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
# embeddings = mean_pooling(outputs[0], inputs['attention_mask'])


def load_contriever():
    print(f"Loading model from: {c_args.model_name_or_path}")
    model, tokenizer, _ = retrieval_contriever.src.contriever.load_retriever(c_args.model_name_or_path)
    return model, tokenizer
    # model.eval()
    # model = model.cuda()
    # if not c_args.no_fp16:
    #     model = model.half()

    # sentences = [
    #     "Where was Marie Curie born?",
    #     "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    #     "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
    # ]

    # # Apply tokenizer
    # inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    # # Compute token embeddings
    # outputs = model(**inputs)
    # score1 = outputs[0] @ outputs[1]
    # score2 = outputs[0] @ outputs[2]
    # print(score1)
    # print(score2)


def load_passages_id_map():
    index = retrieval_contriever.src.index.Indexer(c_args.projection_size, c_args.n_subquantizers, c_args.n_bits)
    # index all passages
    input_paths = glob.glob(c_args.passages_embeddings)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0])
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if c.save_or_load_index and os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        print(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, c_args.indexing_batch_size)
        print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
        if c_args.save_or_load_index:
            index.serialize(embeddings_dir)

    # load passages
    passages = src.data.load_passages(c_args.passages)
    passage_id_map = {x["id"]: x for x in passages}
    return passage_id_map, index

def beam_retrieve(input, contriever_model, contriever_tokenizer, passage_id_map, index):
    queries = [input[0] + input[1]]
        # queries = [ex["question"] for ex in data]
    questions_embedding = embed_queries(c_args, queries, contriever_model, contriever_tokenizer)
    # get top k results
    # start_time_retrieval = time.time()
    top_ids_and_scores = index.search_knn(questions_embedding, c_args.n_docs)
    # print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")
    m_docs = list()
    m_scores = list()
    for i, score in enumerate(top_ids_and_scores):
        docs = [passage_id_map[doc_id] for doc_id in score[0]]
        scores = [str[score] for score in score[1]]
        m_docs.append(docs)
        m_scores.append(scores)
    return m_docs, m_scores

def llama_model_init():
    # contriever, contriever_tokenizer = load_contriever()
    base_model = model_init(args.base_model_path)
    sk_model, sk_tokenizer = model_init(args.self_knowledge_model_path)
    pr_model, pr_tokenizer = model_init(args.passage_relevance_model_path)
    td_model, td_tokenizer = model_init(args.task_decomposition_model_path)
    return Self_Knowledge_Model(sk_model, sk_tokenizer), Passage_Relevance_Model(pr_model, pr_tokenizer), Task_Decomposition_Model(td_model, td_tokenizer)

def problem_solving(input, iter, SKM, PRM, TDM, contriever, contriever_tokenizer, base_model, tokenizer, passage_id_map, index):
    if iter > args.iteration_max_time:
        return "unknow"
    if SKM.find_known(input[0], input[1]):
        prompt = "Give the answer to the question: "
        answer = predict(args, input[0] + prompt + input[1], base_model, tokenizer)
        return answer
    m_docs, m_scores = beam_retrieve(input, contriever, contriever_tokenizer, passage_id_map, index)
    r_docs = []
    for _, docs in enumerate(m_docs):
        for idx, doc in enumerate(docs):
            if PRM.find_relevance(input[0], input[1], doc[idx]["text"]):
                r_docs.append(doc)
    if len(r_docs) > 0:
        ref = ""
        for idx, doc in enumerate(r_docs):
            ref = ref + "\nPragraphs " + str(idx) + ":" + doc["text"]
        ref = ref + "\nUse the knowledge from the relevant paragraphs, give the answer to the question."
        answer = predict(args, input[0] + ref + input[1], base_model, tokenizer)
        return answer
    TDM.decompose()
    sub_qas = []
    for idx, sub_query in enumerate(TDM.query_list):
        sub_answer = problem_solving([input[0], sub_query], iter, SKM, PRM, TDM, contriever, contriever_tokenizer, base_model, tokenizer, passage_id_map, index)
        sub_qa = [sub_query, sub_answer]
        sub_qas.append(sub_qa)
    sub_str = ""
    for idx, sub_qa in enumerate(sub_qas):
        sub_str = sub_str + "\nsub_question " + idx + ": " + sub_qa[0]
        sub_str = sub_str + "\nsub_question " + idx + ": " + sub_qa[1]
    sub_str = sub_str + "\nBase on the sub-question answer. give the answer to the origin question."
    answer = predict(args, input[0] + sub_str + input[1], base_model, tokenizer)
    return answer

def run_llama(dataset, SKM, PRM, TDM, contriever, contriever_tokenizer, base_model, tokenizer, passage_id_map, index):
    answer_set = list()
    for idx, data in enumerate(dataset):
        input = [data['context'], data['query']] 
        ans = problem_solving(input, 0, SKM, PRM, TDM, contriever, contriever_tokenizer, base_model, tokenizer, passage_id_map, index)
        answer_set.append(ans)
    with open(args.output_path, 'a', encoding = "UTF-8") as f:
        for idx, ans in enumerate(answer_set):
            f.write(json.dumps(ans) + '\n')
 

if __name__ == '__main__':
    print_exp(args)
    print_exp(c_args)
    base_model, tokenizer = model_init(args.base_model_path)
    contriever, contriever_tokenizer = load_contriever()
    dataset = load_dataset(args.data_path)
    passage_id_map, index = load_passages_id_map()
    base_model, tokenizer = model_init(args.base_model_path)
    SKM, PRM, TDM = llama_model_init()
    answer = run_llama(dataset, SKM, PRM, TDM, contriever, base_model, passage_id_map, index)
    # print(answer)