# Description: Language model wrapper for RAG
from transformers import AutoTokenizer, pipeline, GenerationConfig
from vllm import LLM, SamplingParams
from abc import ABC, abstractmethod
from datasets import Dataset
import torch, os
from tqdm.auto import tqdm
from argparse import Namespace
from typing import List, Optional
from pydantic import BaseModel
from ast import literal_eval
import re

class AnswerFormat(BaseModel):
    answer: str
    explanation: str

PROMPT_TEMPLATE = {
    "Llama2Chat": "<s>[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{PROMPT} [/INST]",
    "Mistral": "<s>[INST] {PROMPT} [/INST]",
    "Mixtral": "<s>[INST] {PROMPT} [/INST]",
    "Qwen": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n",
    "Gemma": "{PROMPT}",
    "Phi": "{PROMPT}",
    "Orca": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant",
    "GPT": "{PROMPT}"
}

QA_DECODER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string"
        },
        "explanation": {
            "type": "string"
        }
    }
}


    
class LM(ABC):
    def __init__(self, model, mode, norm_name: str = None):
        self.mode = mode
        self.model_name = model
        self.normalized_name = norm_name
        if self.mode == "hf":
            #self.config = GenerationConfig(do_sample=False)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token_id = 2
            self.model = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device_map="auto",
                framework="pt",
                max_new_tokens=200
            )
        elif self.mode == "vllm":
            if self.normalized_name == "GPT":
                self.model = None
                self.config = None
            else:
                self.model = LLM(model=self.model_name,
                                tensor_parallel_size=torch.cuda.device_count(),
                                seed=42)
                self.config = SamplingParams(temperature=.0, max_tokens=100)

    @classmethod
    def load_model(cls, model_name: str, mode: str = "vllm"):
        if "llama" in model_name.lower():
            return cls(model_name, mode, "Llama2Chat")
        elif "qwen" in model_name.lower():
            return cls(model_name, mode, "Qwen")
        elif "mistral" in model_name.lower():
            return cls(model_name, mode, "Mistral")
        elif "mixtral" in model_name.lower():
            return cls(model_name, mode, "Mixtral")
        elif "phi" in model_name.lower():
            return cls(model_name, mode, "Phi")
        elif "orca" in model_name.lower():
            return cls(model_name, mode, "Orca")
        elif "gemma" in model_name.lower():
            return cls(model_name, mode, "Gemma")
        elif "gpt" in model_name.lower():
            return cls(model_name, mode, "GPT")
        elif "gemini" in model_name.lower():
            return cls(model_name, mode, "Gemini")
        else:
            raise ValueError("Model not supported")
    
    def generate(self, dataset: Dataset) -> List[str]:
        results = []
        dataset = _prepare_dataset(dataset, self.normalized_name)
        if self.mode == "hf":
            for out in tqdm(
                self.model(dataset, max_new_tokens=20), desc="Generating", total=len(dataset)
                ):
                results.append(out["generated_text"].strip())
            return results
        elif self.mode == "vllm":
            if self.normalized_name == "GPT":
                for out in tqdm(
                    self.model.generate(dataset["input"], self.config), desc="Generating", total=len(dataset)):
                    results.append(out.strip())
                return results
            else:
                return [o.outputs[0].text.strip() for o in self.model.generate(dataset["input"], self.config)]

    def generate_with_parser(self, dataset: Dataset, parser: str) -> List[str]:
        assert self.mode == "hf", "Parser is only supported for Huggingface Pipelines"
        results = []
        dataset = _prepare_dataset(dataset, self.normalized_name)
        if parser == "json_former":
            from langchain_experimental.llms import JsonFormer
            json_former = JsonFormer(json_schema=QA_DECODER_SCHEMA, pipeline=self.model)
            for prompt in dataset["input"]:
                out = json_former.predict(prompt)
                print(out, type(out))
                out = literal_eval(out)
                results.append(out["answer"].strip())
            return results
        elif parser == "format_enforcer":
            from lmformatenforcer import JsonSchemaParser
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            parser = JsonSchemaParser(AnswerFormat.schema())
            prefix_function = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
            for prompt in tqdm(dataset["input"], desc="Generating"):
                prompt += f"\nYou MUST answer using the following json schema: {str(AnswerFormat.schema_json())}\n\n"
                output_dict = self.model(prompt,
                                         prefix_allowed_tokens_fn=prefix_function,
                                         max_new_tokens=500,
                                         pad_token_id=2)
                result = output_dict[0]['generated_text'][len(prompt):]
                try:
                    result_dict = literal_eval(result)
                    output = result_dict["answer"]
                except:
                    match = re.search(r'"answer":\s*"([^"]*)"', result)
                    if match:
                        output = match.group(1)
                    else:
                        output = "PARSE_ERROR"
                print(output)
                results.append(output)
            return results
        else:
            raise ValueError("Parser not supported")

def _prepare_dataset(dataset: Dataset, normalized_name: str):
    return dataset.map(lambda x: {"input":PROMPT_TEMPLATE[normalized_name].format(PROMPT=x["prompt"])}, num_proc=4)