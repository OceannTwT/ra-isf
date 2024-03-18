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

import numpy as np
import torch
import transformers

class Task_Decomposition_Model():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.query_list = list()

    def decompose(self, context, query):
        inputs = tokenizer(context + query, return_tensors="pt").to('cuda')
        generate_ids = model.generate(**inputs, max_length=512, temperature=args.temperature)
        generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
        result = tokenizer.decode(generate_ids)
        try:
            data = json.loads(result)
            for idx, q in data['query']:
                self.query_list.append(q)
        except json.JSONDecodeError:
            print(f"Invalid format on TDM query: {context + query}, json_string: {result}")
