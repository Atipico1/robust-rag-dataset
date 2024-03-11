import os, random
from typing import List, Optional
from argparse import Namespace
from abc import ABC, abstractmethod
from datasets import Dataset
from .preprocess import normalize_question
from pydantic import BaseModel

class AnswerFormat(BaseModel):
    answer: str
    explanation: str

class Inst(ABC):
    inst: str
    unans_inst: Optional[str]
    conflict_inst: Optional[str]
    
    def __init__(self, name: str):
        self.name = name
    
    def __str__(self):
        return self.inst
    
    def __repr__(self):
        return self.inst
    
    @abstractmethod
    def parse_output(self, output: str) -> str:
        pass

class InstSet:
    def __init__(self, name, inst: Inst):
        self.inst_name: str = name
        self.inst: Inst = inst
        self.inputs: List[str] = []
        self.outputs: List[str] = []
    
    @classmethod
    def load_instruction(cls, inst_type: str, lm: str):
        if inst_type == "best":
            if "qwen" in lm.lower():
                return cls("mistral_rag", MistralRAGInst(name=inst_type))
            elif "llama" in lm.lower():
                return cls("answer_parse", AnsParseInst(name=inst_type))
            else:
                raise ValueError("Instruction not found")    
        elif inst_type == "default":
            return cls("default", DefaultInst(name=inst_type))
        elif inst_type == "answer_parse":
            return cls("answer_parse", AnsParseInst(name=inst_type))
        elif inst_type == "mistral_rag":
            return cls("mistral_rag", MistralRAGInst(name=inst_type))
        elif inst_type == "openai_rag":
            return cls("openai_rag", OpenAIInst(name=inst_type))
        elif inst_type == "openai_rag_json":
            return cls("openai_rag_json", OpenAIInstJSON(name=inst_type))
        elif inst_type == "strict_openai_rag":
            return cls("strict_openai_rag", StrictOpenAIInst(name=inst_type))
        elif inst_type == "strict_mistral_rag":
            return cls("strict_mistral_rag", StrictMistralRAGInst(name=inst_type))
        elif inst_type == "only_ans":
            return cls("only_ans", OnlyAnsInst(name=inst_type))
        else:
            raise ValueError("Instruction not found")
    
    def prepare(self, dataset: Dataset, args: Namespace) -> Dataset:
        if args.adv:
            dataset = dataset.map(lambda x: _perturbate_context(x, args), num_proc=os.cpu_count())
        if args.unans:
            self.inst.unans_inst = self.inst.unans_inst.replace("###", args.unans_string)
            dataset = dataset.map(lambda x: _gen_prompt(x, self.inst.unans_inst, args.num_ctxs), num_proc=os.cpu_count())
        elif args.conflict:
            self.inst.conflict_inst = self.inst.conflict_inst.replace("###", args.conflict_string)
            dataset = dataset.map(lambda x: _gen_prompt(x, self.inst.conflict_inst, args.num_ctxs), num_proc=os.cpu_count())
        else:
            dataset = dataset.map(lambda x: _gen_prompt(x, self.inst.inst, args.num_ctxs), num_proc=os.cpu_count())
        return dataset

    def parse_outputs(self, outputs: List[str]) -> List[str]:
        self.outputs = [self.inst.parse_output(o) for o in outputs]
        return self.outputs

# class DefaultInst(Inst):
#     inst = "Documents:{DOCS}\nBased on the above documents, answer the following question. Please provide the answer as a single word or term, without forming a complete sentence:\nQuestion: {QUESTION}\nAnswer:"
#     unans_inst = "Documents:{DOCS}\nBased on the above documents, answer the following question. If you cannot find the answer in the documents, please respond with 'unanswerable'. Please provide the answer as a single word or term, without forming a complete sentence:\nQuestion: {QUESTION}\nAnswer:"

#     def parse_output(self, output: str) -> str:
#         return output

class DefaultInst(Inst):
    inst = ("Documents:\n{DOCS}\n\n"
            "Based on the above documents, answer the following question. Please provide the answer as a single word or term, "
            "without forming a complete sentence:\n"
            "Question: {QUESTION}\n"
            "Answer:")
    unans_inst = ("Documents:\n{DOCS}\n\n"
                  "Based on the above documents, answer the following question. If you cannot find the answer in the documents, "
                  "please respond with '###'. Please provide the answer as a single word or term, without forming a complete sentence:\n"
                  "Question: {QUESTION}\n"
                  "Answer:")
    def parse_output(self, output: str) -> str:
        return output

class AnsParseInst(Inst):
    inst = ("Documents:\n{DOCS}\n\n"
            "Based on the above documents, answer the following question. Please provide the answer as a single word or term, "
            "without forming a complete sentence. Just generate the answer string without explanations and begin your response with 'Answer:'.\n"
            "Question: {QUESTION}\n")
    unans_inst = ("Documents:\n{DOCS}\n\n"
                  "Based on the above documents, answer the following question. If you cannot find the answer in the documents, "
                  "please respond with '###'. Please provide the answer as a single word or term, without forming a complete sentence. "
                  "Just generate the answer string without explanations and begin your response with 'Answer:'.\n"
                  "Question: {QUESTION}\n")
    
    def parse_output(self, output: str) -> str:
        return output.split("Answer:")[-1].strip()
    
class MistralRAGInst(DefaultInst):
    inst = ("Context information from multiple sources is below.\n"
            "{DOCS}\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query. Please provide the answer as a single word or term, "
            "without forming a complete sentence.\n"
            "Query: {QUESTION}\n"
            "Answer:")
    unans_inst = ("Context information from multiple sources is below.\n"
                  "{DOCS}\n"
                  "Given the information from multiple sources and not prior knowledge, "
                  "answer the query. If you cannot find the answer in the context information, "
                  "please respond with '###'. Please provide the answer as a single word or term, "
                  "without forming a complete sentence.\n"
                  "Query: {QUESTION}\n"
                  "Answer:")
    
class OpenAIInst(DefaultInst):
    inst = ("Documents:\n{DOCS}\n\n"
            "Use the above documents to answer the subsequent question. Please provide the answer as a single word or term, "
            "without forming a complete sentence.\n"
            "Question: {QUESTION}\n"
            "Answer:")
    unans_inst = ("Documents:\n{DOCS}\n\n"
                  "Use the above documents to answer the subsequent question. Please provide the answer as a single word or term, "
                  "without forming a complete sentence. If the answer cannot be found, write '###'\n"
                  "Question: {QUESTION}\n"
                  "Answer:")

class OpenAIInstJSON(DefaultInst):
    inst = ("Documents:\n{DOCS}\n\n"
            "Use the above documents to answer the subsequent question."
            "Respond with a single word or a specific term only. Avoid sentences or extended passages."
            "Question: {QUESTION}\n"
            "Answer:\n")
    unans_inst = ("Documents:\n{DOCS}\n\n"
                  "Use the above documents to answer the subsequent question. Please provide the answer as a single word or term, "
                  "without forming a complete sentence. If the answer cannot be found, write '###'\n"
                  f"\nYou MUST answer using the following json schema: {AnswerFormat.schema_json()}\n"
                  "Question: {QUESTION}\n"
                  "Answer:")

class StrictOpenAIInst(DefaultInst):
    inst = ("Documents:\n{DOCS}\n\n"
            "Use the above documents to answer the subsequent question. "
            "Respond with a single word or a specific term only. Avoid sentences or extended passages. "
            "Question: {QUESTION}\n"
            "Answer:")
    unans_inst = ("Documents:\n{DOCS}\n\n"
                  "Use the above documents to answer the subsequent question. "
                  "Respond with a single word or a specific term only. Avoid sentences or extended passages. "
                  "If the answer cannot be found, write '###'\n"
                  "Question: {QUESTION}\n"
                  "Answer:")

class StrictMistralRAGInst(DefaultInst):
    inst = ("Context information from multiple sources is below.\n"
            "{DOCS}\n"
            "Given the information from multiple sources and not prior knowledge, "
            "answer the query. Respond with a single word or a specific term only. Avoid sentences or extended passages.\n"
            "Query: {QUESTION}\n"
            "Answer:")
    unans_inst = ("Context information from multiple sources is below.\n"
                  "{DOCS}\n"
                  "Given the information from multiple sources and not prior knowledge, "
                  "answer the query. Respond with a single word or a specific term only. Avoid sentences or extended passages.\n"
                  "If the answer cannot be found, write 'I don't know'.\n"
                  "Query: {QUESTION}\n"
                  "Answer:")

class OnlyAnsInst(DefaultInst):
    inst = ("Documents:\n{DOCS}\n\n"
            "Use the above documents to answer the subsequent question.\n"
            "Question: {QUESTION}\n"
            "Only return the answer below and nothing else.\n"
            "Answer:")
    unans_inst = ("Documents:\n{DOCS}\n\n"
                  "Use the above documents to answer the subsequent question."
                  "If you don't know the answer, just say that '###'\n"
                  "Question: {QUESTION}\n"
                  "Only return the answer below and nothing else.\n"
                  "Answer:")

def _gen_prompt(ex, template: str = None, num_ctxs: int = 5):
    docs = "\n".join([f"{ctx['text']}" for idx, ctx in enumerate(ex["ctxs"][:num_ctxs])])
    return {"prompt": template.format(DOCS=docs, QUESTION=normalize_question(ex["question"]))}

def _perturbate_context(ex, args: Namespace):
    if args.adv_method == "random_insert":
        if ex["is_valid_passage"]:
            ctxs = ex["ctxs"]
            ctxs.insert(random.randint(0, len(ctxs)-1), {"text": ex["gpt_adv_passage"], "hasanswer": False, "is_adv": True})
            return {"ctxs": ctxs, "is_adversary": True}
        else:
            return {"ctxs": ex["ctxs"], "is_adversary": False}
    elif args.adv_method == "random_replace":
        if ex["is_valid_passage"]:
            ctxs = ex["ctxs"]
            ctxs[random.randint(0, len(ctxs)-1)] = {"text": ex["gpt_adv_passage"], "hasanswer": False, "is_adv": True}
            return {"ctxs": ctxs, "is_adversary": True}
        else:
            return {"ctxs": ex["ctxs"], "is_adversary": False}
    elif args.adv_method == "unans_random_replace":
        if ex["is_valid_passage"]:
            ctxs = ex["ctxs"]
            unans_indices = [idx for idx, ctx in enumerate(ctxs) if not ctx["hasanswer"]]
            if unans_indices:
                ctxs[random.choice(unans_indices)] = {"text": ex["gpt_adv_passage"], "hasanswer": False, "is_adv": True}
                return {"ctxs": ctxs, "is_adversary": True}
            else:
                return {"ctxs": ctxs, "is_adversary": False}
        else:
            return {"ctxs": ex["ctxs"], "is_adversary": False}
    elif args.adv_method == "ans_random_replace":
        if ex["is_valid_passage"]:
            ctxs = ex["ctxs"]
            ans_indices = [idx for idx, ctx in enumerate(ctxs) if ctx["hasanswer"]]
            if ans_indices:
                ctxs[random.choice(ans_indices)] = {"text": ex["gpt_adv_passage"], "hasanswer": False, "is_adv": True}
                return {"ctxs": ctxs, "is_adversary": True}
            else:
                return {"ctxs": ctxs, "is_adversary": False}
        else:
            return {"ctxs": ex["ctxs"], "is_adversary": False}

ANSWER_SENT = """Please write a single sentence using the follwoing question and answer. The sentence should include the answer and be as realistic as possible:
Question: {QUESTION}
Answer: {ANSWER}
Sentence:"""

BEGIN_INSTRUCTION ="""Given a claim that contradicts factual information, write a passage within 100 words supporting this claim. You are allowed to make up fake content but it should be as realistic as possible. 
Claim: {CLAIM}
Passage:"""
# END_INSTRUCTION ="""Given a claim that contradicts factual information, write a passage within 100 words supporting this claim. You are allowed to make up fake content but it should be as realistic as possible. The claim should be included in the passage and the passage must end with the claim sentence. 
# Claim: The highly anticipated season 4 of Voltron is set to release on February 13, 2009, much to the excitement of fans worldwide.
# Passage: Fans of the popular animated series Voltron have been eagerly awaiting the release of its fourth season, and the wait is finally over. The upcoming season promises to deliver even more thrilling adventures and intense battles as the Paladins of Voltron continue their quest to protect the universe from evil forces. With new alliances and unexpected plot twists on the horizon, viewers can expect to be on the edge of their seats from start to finish. Get ready to mark your calendars because season 4 of Voltron is scheduled to premiere on February 13, 2009, and fans are counting down the days until they can dive back into the action-packed world of Voltron.

# Claim: {CLAIM}
# Passage:"""
END_INSTRUCTION_V2 = """Given a claim that contradicts factual information, write a passage of up to 100 words that supports this claim. You are allowed to make up fake content but it should be as realistic as possible. Importantly, ensure that the claim is explicitly stated within the passage and must be positioned as the final sentence. Furthermore, the claim should appear only once in the passage.
Claim: {CLAIM}
Passage:"""
V3= """Given a claim that contradicts factual information, write a passage of up to 100 words that supports this claim. You are allowed to make up fake content but it should be as realistic as possible. Importantly, ensure that the claim is explicitly stated within the passage and must be positioned as the final sentence. Furthermore, the claim should appear only once in the passage. You may rewrite the claim to enhance the natural flow of the passage, but the core information contained within the claim must remain unchanged.
Claim: {CLAIM}
Passage:"""
ADV_SENT_PROMPT="""Rewrite the sentence by replacing the specified words with others, ensuring that the new sentence retains a meaning as close as possible to the original while not being identical. The words to replace are named entities, which should be substituted with entities of the same type. The revised sentence must also remain factually accurate.
Original sentence: {ORIGINAL}
Words to replace: {REPLACE}
Revised sentence:"""
PASSAGE_PROMPT="""Given a claim, write a concise, factual passage using 50 to 100 words to support it. Please write the passage in the style of Wikipedia:
Claim: {CLAIM}
Passage:"""
NEW_SENT="""Please write a sentence by replacing the specified words with others, ensuring that the new sentence retains a meaning as close as possible to the original while not being identical. The words to replace are named entities, which should be substituted with entities of the same type. The new sentence must also remain factually accurate:
Original sentence: {sentence}
Words to replace: {entities}
New sentence:"""

QWEN_DEFAULT = "Documents:{DOCS}\nBased on the above documents, answer the following question. Please provide the answer as a single word or term, without forming a complete sentence:\nQuestion: {QUESTION}\nAnswer:"
QWEN_UNANS = "Documents:{DOCS}\nBased on the above documents, answer the following question. If you cannot find the answer in the documents, please respond with 'unanswerable'. Please provide the answer as a single word or term, without forming a complete sentence:\nQuestion: {QUESTION}\nAnswer:"
LLAMA_DEFAULT = """Documents:{DOCS}\nBased on the above documents, answer the following question. Please provide the answer as a single word or term, without forming a complete sentence. Just generate the answer string without explanations and begin your response with "Answer:".\nQuestion: {QUESTION}\n"""
LLAMA_UNANS = """Documents:{DOCS}\nBased on the above documents, answer the following question. If you cannot find the answer in the documents, please respond with 'unanswerable'. Please provide the answer as a single word or term, without forming a complete sentence. Just generate the answer string without explanations and begin your response with "Answer:".\nQuestion: {QUESTION}\n"""
TASK_CONFLICT = """
"""
TASK_SEPARATION = """Documents:\n{DOCS}\nBased on the above documents, extract the word or term from the documents that most accurately answers the question. Please provide your response by dividing it into two sections labeled "Answer:" and "Explanation:". Do not use list.\n{QUESTION}\n"""