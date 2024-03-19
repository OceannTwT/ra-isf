python main.py --engine "llama2-13b"
      --base_model_path {base_model_path} \
      --self_knowledge_model_path {self_knowledge_model_path} \
      --passage_relevance_model_path {passage_relevance_model_path} \
      --task_decomposition_model_path {task_decomposition_model_path} \
      --data_path {data_path} \
      --n_docs {Number of documents to retrieve per questions} \
      --model_name_or_path {contriever_model_path} \
      --passages_embedding "wikipedia_embeddings/*" \