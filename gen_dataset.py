from argparse import ArgumentParser, Namespace
from utils.preprocess import normalize_question
import os, wandb
from utils.instruction import *
from datasets import load_dataset, Dataset
import pandas as pd
from utils.util import str2bool
from utils.preprocess import has_answer, SimpleTokenizer

def _gen_prompt(ex, template: str = None):
    docs = "\n".join([f"{ctx['text']}" for idx, ctx in enumerate(ex["ctxs"][:5])])
    return {"prompt": template.format(DOCS=docs, QUESTION=normalize_question(ex["question"]))}

def gen_prompt(dataset: Dataset, template):
    dataset = dataset.map(lambda x: _gen_prompt(x, template), num_proc=os.cpu_count())
    return dataset

def gen_conflict(dataset: Dataset, args: Namespace):
    pass

def gen_adversary(dataset: Dataset, args: Namespace):
    pass

def gen_unans(dataset: Dataset, args: Namespace):
    dataset = dataset.map(lambda x: {"hasanswer": any([ctx["hasanswer"] for ctx in x["ctxs"]])}, num_proc=os.cpu_count())
    dataset = dataset.map(lambda x: {"answers": [args.unans_string] if not x["hasanswer"] else x["answers"]}, num_proc=os.cpu_count())
    return dataset

def save_samples(dataset: Dataset, args: Namespace):
    wandb.init(
        project="generate-dataset", name=args.output_path if not args.test else f"test-{args.output_path}", config=vars(args)
        )
    df = pd.DataFrame(dataset.shuffle(42).select(range(100)))
    df = df[["question", "answers", "prompt"]]
    wandb.log({"samples": wandb.Table(dataframe=df)})

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--origin_path", type=str, default="Atipico1/NQ")
    parser.add_argument("--output_path", type=str, default="Atipico1/NQ")
    parser.add_argument("--num_contexts", type=int, default=5)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--task", type=str, nargs="+", default="")
    parser.add_argument("--test", type=str2bool, default=False)
    args: Namespace = parser.parse_args()
    dataset = load_dataset(args.origin_path, split=args.split)
    if "conflict" in args.task:
        pass
    if "adversary" in args.task:
        pass
    if "unans" in args.task:
        dataset = gen_unans(dataset, args)
    dataset = gen_prompt(dataset)
    save_samples(dataset, args)
    dataset.push_to_hub(args.output_path)