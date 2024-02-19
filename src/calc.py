import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from datasets import Dataset
from collections import Counter
from utils.preprocess import normalize_answer, SimpleTokenizer, has_answer
from typing import List, Callable

def caluclate_metrics(outputs: List[str], dataset: Dataset):
    tokenizer = SimpleTokenizer()
    ems = [exact_match_score(pred, label) for pred, label in zip(outputs, dataset["answers"])]
    f1s = [f1_score(pred, label) for pred, label in zip(outputs, dataset["answers"])]
    accs = [has_answer(label, pred, tokenizer) for label, pred in zip(dataset["answers"], outputs)]
    print(f"EM: {sum(ems)/len(ems)}, F1: {sum(f1s)/len(f1s)}, ACC: {sum(accs)/len(accs)}")
    
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