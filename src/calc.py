import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from datasets import Dataset
from collections import Counter
from utils.preprocess import normalize_answer, SimpleTokenizer, has_answer
from typing import List, Callable, Tuple
import pandas as pd
import wandb

def caluclate_metrics(outputs: List[str], dataset: Dataset, raw_outputs: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tokenizer = SimpleTokenizer()
    ems = [bool(exact_match_score(pred, label)) for pred, label in zip(outputs, dataset["answers"])]
    f1s = [bool(f1_score(pred, label)) for pred, label in zip(outputs, dataset["answers"])]
    accs = [has_answer(label, pred, tokenizer) for label, pred in zip(dataset["answers"], outputs)]
    hasanswer = [any([ctx["hasanswer"] for ctx in ctxs]) for ctxs in dataset["ctxs"]]
    answers = [",".join(ans) for ans in dataset["answers"]]
    df = pd.DataFrame({"EM": ems, "F1": f1s, "Acc": accs, "Prediction": outputs, "hasanswer": hasanswer, "answers": answers})
    df["prompt"] = dataset["prompt"]
    df["raw_output"] = raw_outputs
    em_ans, em_unans = df[df["hasanswer"] == True]["EM"].mean(), df[df["hasanswer"] == False]["EM"].mean()
    acc_ans, acc_unans = df[df["hasanswer"] == True]["Acc"].mean(), df[df["hasanswer"] == False]["Acc"].mean()
    data = df[["EM", "Acc", "F1", "hasanswer"]].mean().to_dict()
    data.update({"EM (ans)": em_ans, "EM (unans)": em_unans, "Acc (ans)": acc_ans, "Acc (unans)": acc_unans})
    data = {k:round(v*100,2) for k,v in data.items()}
    return df, pd.DataFrame(index=[0], data=data)

def em(prediction, ground_truth, normalize_fn: Callable):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))

def f1(prediction, ground_truth, normalize_fn: Callable):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_score(prediction, ground_truths):
    return max([f1(prediction, gt, normalize_answer) for gt in ground_truths])

def exact_match_score(prediction, ground_truths):
    return max([em(prediction, gt, normalize_answer) for gt in ground_truths])