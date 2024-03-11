from datasets import Dataset
from typing import Callable
from .preprocess import normalize_question
import os

def preprocess_dataset(dataset: Dataset) -> Dataset:
    def sub_fn(answers):
        if isinstance(answers, list):
            return answers
        else:
            return [answers]
    dataset = dataset.map(lambda x: {"question": normalize_question(x["question"])}, num_proc=os.cpu_count())
    dataset = dataset.map(lambda x: {"answers": sub_fn(x["answers"])}, num_proc=os.cpu_count())
    return dataset

def filter_and_map(dataset: Dataset,
                   filter_fn: Callable,
                   map_fn: Callable,
                   out_col: str,
                   num_proc: int = min(16, os.cpu_count())) -> Dataset:
    result = []
    dataset = dataset.add_column("temp_id", range(len(dataset)))
    sub_dataset = dataset.filter(filter_fn, num_proc=num_proc)
    sub_dataset = sub_dataset.map(map_fn, num_proc=num_proc)
    for i in range(len(dataset)):
        if i not in sub_dataset["temp_id"]:
            result.append(None)
        else:
            idx = sub_dataset["temp_id"].index(i)
            result.append(sub_dataset[out_col][idx])
    dataset = dataset.add_column(out_col, result)
    dataset = dataset.remove_columns("temp_id")
    return dataset

def ner(dataset: Dataset, in_col: str, out_col: str = "entities") -> Dataset:
    import spacy
    if spacy.prefer_gpu():
        nlp = spacy.load("en_core_web_trf")
    else:
        raise Exception("No GPU found")
    docs = list(nlp.pipe(dataset[in_col]))
    ents, ents_count = [],[]
    for doc in docs:
        if doc.ents:
            ents.append(", ".join(['"'+ent.text+'"' for ent in doc.ents]))
            ents_count.append(len(doc.ents))
        else:
            ents.append(None)
            ents_count.append(0)
    dataset = dataset.add_column(out_col, ents)
    dataset = dataset.add_column(out_col+"_count", ents_count)
    return dataset