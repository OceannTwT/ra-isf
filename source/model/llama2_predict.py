import torch
import time
import os
import json
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoConfig

def model_init(model_path):
    # model_path = args.model_path 
    device = torch.device("cuda:0")
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=False)
    return model, tokenizer

def predict(args, prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    generate_ids = model.generate(**inputs, max_length=args.max_length, temperature=args.temperature)
    generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
    infer_res = tokenizer.decode(generate_ids)
    return infer_res
