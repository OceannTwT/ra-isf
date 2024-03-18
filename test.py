import torch
from transformers import AutoTokenizer
from retrieval_contriever.src.contriever import Contriever

tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/contriever-msmarco')
model = Contriever.from_pretrained('/root/autodl-tmp/contriever-msmarco')

sentences = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, 111111 Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

# Apply tokenizer
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
outputs = model(**inputs)

# Mean pooling
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings
# embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
embeddings = outputs
# print(embeddings[0])
# print(embeddings[1])
score1 = embeddings[0] @ embeddings[1]
score2 = embeddings[0] @ embeddings[2]
print(score1)
print(score2)