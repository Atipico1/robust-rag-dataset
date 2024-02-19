from argparse import ArgumentParser, Namespace
from utils.preprocess import normalize_question
import os, wandb
from utils.prompt import TASK_DEFAULT
from datasets import load_dataset, Dataset
import pandas as pd
from utils.util import str2bool

def _gen_prompt(ex):
    prompt = TASK_DEFAULT
    prompt += "\n".join([f"Doc {idx}: {ctx['text']}" for idx, ctx in enumerate(ex["ctxs"][:args.num_contexts])])
    prompt += f"\nQuestion: {normalize_question(ex['question'])}\nAnswer:"
    return {"prompt":prompt}

def gen_prompt(dataset: Dataset, args: Namespace):
    dataset = dataset.map(_gen_prompt, num_proc=os.cpu_count())
    return dataset

def gen_conflict(dataset: Dataset, args: Namespace):
    pass

def gen_adversary(dataset: Dataset, args: Namespace):
    pass

def gen_unans(dataset: Dataset, args: Namespace):
    passß

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
        pass
    if "default" in args.task:
        dataset = gen_prompt(dataset, args)
    save_samples(dataset, args)
    dataset.push_to_hub(args.output_path)