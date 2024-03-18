import pandas as pd
import datasets

from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from typing import List, Union


def clean_text(
    texts: List[Union[str, None]], labels: List[Union[str, None]]
) -> pd.DataFrame:
    """
    The News Group dataset needs to be preprocessed as it has a lot of
    entries with NULL text and/or NULL labels.
    In this function we simply filter out the NULL entries, and
    return a new dataframe with clean texts and labels.
    """
    new_texts, new_labels = [], []
    for text, label in zip(texts, labels):
        if isinstance(text, str) and isinstance(label, str):
            new_texts.append(text)
            new_labels.append(label)
    new_ids = [i for i in range(len(new_texts))]
    df = pd.DataFrame(data={"id": new_ids, "text": new_texts, "label": new_labels})

    return df

def get_data(tokenizer: AutoTokenizer) -> List[Union[DatasetDict, int, int]]:
    dataset_id = "nq_open"
    # Load dataset from the hub
    dataset = load_dataset(dataset_id)

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['validation'])}") # if validate

    tokenized_inputs = concatenate_datasets([dataset["train"]]).map(
        lambda x: tokenizer(x["question"], truncation=True),
        batched=True,
        remove_columns=["question", "answer"],
    )

    max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
    print(f"Max source length: {max_source_length}")

    tokenized_targets = concatenate_datasets([dataset["train"], dataset["validation"]]).map(
        lambda x: tokenizer(x["answer"], truncation=True),
        batched=True,
        remove_columns=["question", "answer"],
    )

    max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
    print(f"Max target length: {max_target_length}")

    return dataset, max_source_length, max_target_length
