import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iteration_max_time", type=int, default=3, help="maxinum iteration in RA-iSF."
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help=""
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="maxinum generation of base model"
    )
    parser.add_argument(
        "--type_list_file", default="./src/format/entity_type_list.txt", type=str, help='file path'
    )
    parser.add_argument(
        "--prompt_id", default='324', help='string'
    )
    parser.add_argument(
        "--infer_num", default='5', help='string'
    )
    parser.add_argument(
        "--engine", default='llama2-13b', help="llama2-7b, llama2-13b, gpt-3.5",
        choices=["llama2-7b", "llama2-13b", "gpt-3.5"]
    )
    parser.add_argument(
        "--api_key", default="", help="gpt3.5 api key"
    )
    parser.add_argument(
        "--base_model_path", default='/root/autodl-tmp/llama-7b-hf', help="your local model path"
    )
    parser.add_argument(
        "--self_knowledge_model_path", default='/root/autodl-tmp/llama-7b-hf', help="submodel self-knowledge path"
    )
    parser.add_argument(
        "--passage_relevance_model_path", default='/root/autodl-tmp/llama-7b-hf', help="submodel passage_relevance path"
    )
    parser.add_argument(
        "--task_decomposition_model_path", default='/root/autodl-tmp/llama-7b-hf', help="submodel task_decomposition path"
    )
    parser.add_argument(
        "--data_path", default='/root/workspace/ra-isf/dataset/natural_question/nq_open.json', help="your local data path"
    )
    parser.add_argument(
        "--output_path", default='/root/workspace/ra-isf/output/output.json', help="your local output file data path"
    )
    parser.add_argument(
        "--test_start", default='0', help='string, number'
    )
    parser.add_argument(
        "--test_end", default='full', help='string, number'
    )
    parsed_args = parser.parse_args()
    return parsed_args


args = parse_arguments()
