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

class Self_Knowledge_Model():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def find_known(self, context, query):
        inputs = tokenizer(context + query, return_tensors="pt").to('cuda')
        generate_ids = model.generate(**inputs, max_length=512, temperature=args.temperature)
        generate_ids = generate_ids[0][len(inputs["input_ids"][0]):-1]
        result = tokenizer.decode(generate_ids)
        if result == "know":
            return True
        elif result == "unknow":
            return False
        else:
            print(f"Invalid output on SKM query: {context + query}")
            return False
