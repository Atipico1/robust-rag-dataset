from argparse import ArgumentParser, Namespace
import os, torch
from utils.util import str2bool
from models.lm import LM
from datasets import Dataset, load_dataset
from src.calc import caluclate_metrics
def main(args: Namespace):
    model = LM.load_model(args)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.test:
        dataset = dataset.select(range(100))
    output = model.generate(dataset)
    caluclate_metrics(output, dataset)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B-Chat")
    parser.add_argument("--dataset", type=str, default="Atipico1/NQ")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--mode", type=str, default="vllm")
    parser.add_argument("--test", type=str2bool, default=False)
    print("Available GPUs:", torch.cuda.device_count())
    args: Namespace = parser.parse_args()
    main(args)