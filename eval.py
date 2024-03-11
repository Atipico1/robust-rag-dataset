from argparse import ArgumentParser, Namespace
import os, torch
from utils.util import NORM_MAPPING, save_metrics_to_wandb, str2bool, save_output_to_wandb
from models.lm import LM
from utils.instruction import *
from datasets import load_dataset
from src.calc import caluclate_metrics
from gen_dataset import gen_unans
import pandas as pd

FAMILY_MAP = {
    "qwen": ["Qwen/Qwen1.5-7B-Chat", "Qwen/Qwen1.5-14B-Chat", "Qwen/Qwen1.5-72B-Chat"],
    "llama": ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf", "meta-llama/Llama-2-70b-chat-hf"],
    "mistral": ["mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mixtral-8x7B-Instruct-v0.1"],
    "phi": ["microsoft/phi-2"],
    "orca": ["microsoft/Orca-2-13b", "microsoft/Orca-2-7b"],
    "gemma": ["google/gemma-7b-it", "google/gemma-2b-it"],
}

def main(args: Namespace):
    model = LM.load_model(args)
    dataset = load_dataset(args.dataset, split=args.split)
    if args.test:
        dataset = dataset.select(range(100))
    output = model.generate(dataset)
    raw_output, metrics = caluclate_metrics(output, dataset)
    save_output_to_wandb(raw_output, metrics, args)

def parse_test(raw_output: str):
    splited_text = raw_output.split("\n")
    for text in splited_text:
        if text.startswith("Answer:"):
            return text.split("Answer:")[-1].strip()
    return "No answer found"

def prompt_test(model: LM, instruction_set: InstSet, args: Namespace):
    dataset = load_dataset(args.dataset, split=args.split)
    if args.test:
        dataset = dataset.shuffle(seed=42).select(range(args.test_size))
    if args.unans:
        dataset = gen_unans(dataset, args)
    dataset = instruction_set.prepare(dataset, args)
    preds = model.generate(dataset)
    #preds = model.generate_with_parser(dataset, args.parser)
    output = instruction_set.parse_outputs(preds)
    raw_output, metrics = caluclate_metrics(output, dataset, preds)
    args.inst_type = instruction_set.inst.name
    if len(args.insts) > 1:
        metrics["instruction"] = [args.inst_type]
        metrics["model"] = [NORM_MAPPING[args.model]]
        raw_output["instruction"] = [args.inst_type]*len(raw_output)
        raw_output["model"] = [NORM_MAPPING[args.model]]*len(raw_output)
        return metrics, raw_output
    else:
        save_output_to_wandb(raw_output, metrics, args)
        return metrics

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen1.5-7B-Chat")
    parser.add_argument("--family", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="Atipico1/NQ")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--mode", type=str, default="vllm")
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--wandb_project", type=str, default="evaluate-llm")
    
    parser.add_argument("--num_ctxs", type=int, default=5)
    
    parser.add_argument("--inst", type=str, default="separation")
    parser.add_argument("--insts", type=str, nargs="+", default=["default"])
    
    parser.add_argument("--unans", type=str2bool, default=False)
    parser.add_argument("--unans_string", type=str, default="I don't know.")
    
    parser.add_argument("--adv", type=str2bool, default=False)
    parser.add_argument("--adv_method", type=str, default="replace")
    
    parser.add_argument("--conflict", type=str2bool, default=False)
    parser.add_argument("--conflict_string", type=str, default="I don't know.")
    
    parser.add_argument("--parser", type=str, default="json_former")
    print("Available GPUs:", torch.cuda.device_count())
    args: Namespace = parser.parse_args()
    if args.family:
        result = []
        for model_name in FAMILY_MAP[args.family]:
            args.model = model_name
            model = LM.load_model(model_name, args.mode)
            for inst_type in args.insts:
                instruction_set = InstSet.load_instruction(inst_type, model_name)
                output: pd.DataFrame = prompt_test(model, instruction_set, args)
                if result is None:
                    result = output
                else:
                    result = pd.concat([result, output])
        save_metrics_to_wandb(result, args)
    else:
        result, raw_result = None, None
        model = LM.load_model(args.model, args.mode)
        for inst_type in args.insts:
            instruction_set = InstSet.load_instruction(inst_type, args.model)
            output, raw_output = prompt_test(model, instruction_set, args)
            if result is None:
                result = output
                raw_result = raw_output
            else:
                result = pd.concat([result, output])
                raw_result = pd.concat([raw_result, raw_output])
        save_output_to_wandb(raw_result, result, args)
    # main(args)