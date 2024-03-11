from argparse import ArgumentTypeError, Namespace
import pandas as pd
import wandb
from .instruction import *

NORM_MAPPING = {
    "Qwen/Qwen1.5-7B-Chat": "Qwen-7B",
    "Qwen/Qwen1.5-14B-Chat": "Qwen-14B",
    "Qwen/Qwen1.5-72B-Chat": "Qwen-72B",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral",
    "meta-llama/Llama-2-7b-chat-hf": "Llama-7B",
    "meta-llama/Llama-2-13b-chat-hf": "Llama-13B",
    "meta-llama/Llama-2-70b-chat-hf": "Llama-70B",
    "microsoft/Orca-2-13b": "Orca-13B",
    "microsoft/Orca-2-7b": "Orca-7B",
    "microsoft/phi-2": "Phi-2",
    "google/gemma-7b-it": "Gemma-7B",
    "google/gemma-2b-it": "Gemma-2B",
    "gpt-4-0125-preview": "GPT-4",
    "gpt-3.5-turbo-0125": "GPT-3.5",
}

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')

def extract_name(origin_path: str) -> str:
    if "nq" in origin_path.lower():
        if "replace" in origin_path.lower():
            return "NQ-ADV"
        else:
            return "NQ"
    if "trivia" in origin_path.lower():
        return "TQA"

def match_prompt(args: Namespace) -> str:
    model_name = args.model.lower()
    if "qwen" in model_name:
        if args.unans:
            return QWEN_UNANS
        return QWEN_DEFAULT
    elif "mistral" in model_name:
        return "mistral_default"
    elif "llama" in model_name:
        if args.unans:
            return LLAMA_UNANS
        return LLAMA_DEFAULT

def save_output_to_wandb(raw_output: pd.DataFrame, metrics: pd.DataFrame, args: Namespace):
    run_name = f"{NORM_MAPPING[args.model]}||{extract_name(args.dataset)}"
    if args.test:
        run_name = "test-"+run_name
    if args.unans:
        run_name = run_name+"-unans"
    if "random" in args.model:
        run_name = run_name+"-random"
    #run_name = f"{run_name}||{args.inst_type}"
    wandb.init(
        project=args.wandb_project, name=run_name, config=vars(args)
        )
    wandb.log({"raw output": wandb.Table(dataframe=raw_output.sample(1000))})
    wandb.log({"metrics": wandb.Table(dataframe=metrics)})
    wandb.finish()

def save_metrics_to_wandb(metrics: pd.DataFrame, args: Namespace):
    run_name = f"{NORM_MAPPING[args.model]}||{extract_name(args.dataset)}"
    if args.test:
        run_name = "test-"+run_name
    if args.unans:
        run_name = run_name+"-unans"
    if "random" in args.model:
        run_name = run_name+"-random"
    wandb.init(
        project=args.wandb_project, name=run_name, config=vars(args)
        )
    wandb.log({"metrics": wandb.Table(dataframe=metrics)})
    wandb.finish()

