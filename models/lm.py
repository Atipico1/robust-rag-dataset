# Description: Language model wrapper for RAG
from transformers import AutoTokenizer, pipeline, GenerationConfig
from vllm import LLM, SamplingParams
from abc import ABC, abstractmethod
from datasets import Dataset
import torch, os
from tqdm.auto import tqdm
from argparse import Namespace

PROMPT_TEMPLATE = {
    "Llama2Chat": "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{PROMPT} [/INST]",
    "Mistral": "<s>[INST] {PROMPT} [/INST]",
    "Qwen": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"
}

class LM(ABC):
    def __init__(self, model, mode, norm_name: str = None):
        self.mode = mode
        self.model_name = model
        self.normalized_name = norm_name
        if self.mode == "hf":
            self.config = GenerationConfig(do_sample=False)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                generation_config=self.config,
                framework="pt"
            )
        elif self.mode == "vllm":
            self.model = LLM(model=self.model_name,
                             tensor_parallel_size=torch.cuda.device_count(),
                             seed=42)
            self.config = SamplingParams(temperature=.0)

    @classmethod
    def load_model(cls, args: Namespace):
        if "llama" in args.model.lower():
            return cls(args.model, args.mode, "Llama2Chat")
        elif "qwen" in args.model.lower():
            return cls(args.model, args.mode, "Qwen")
        elif "mistral" in args.model.lower():
            return cls(args.model, args.mode, "Mistral")
        else:
            raise ValueError("Model not supported")
    
    def generate(self, dataset: Dataset):
        results = []
        dataset = _prepare_dataset(dataset, self.normalized_name)
        if self.mode == "hf":
            for out in tqdm(
                self.model(dataset, max_new_tokens=20), desc="Generating", total=len(dataset)):
                results.append(out["generated_text"].strip())
            return results
        elif self.mode == "vllm":
            return [o.outputs[0].text.strip() for o in self.model.generate(dataset["input"], self.config)]


def _prepare_dataset(dataset: Dataset, normalized_name: str):
    return dataset.map(lambda x: {"input":PROMPT_TEMPLATE[normalized_name].format(PROMPT=x["prompt"])}, num_proc=4)