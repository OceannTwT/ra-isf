import torch
import time
import os
import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def model_init(args):
    model_path = args.model_path
    device = torch.device("cuda:0")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, device


def predict(args, prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    generate_ids = model.generate(**inputs, temperature=args.temperature)
    generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
    infer_res = tokenizer.decode(generate_ids)
    return infer_res
