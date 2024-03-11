from openai import OpenAI
import pandas as pd
from tqdm.auto import tqdm
import argparse, time
from datasets import load_dataset
from utils.preprocess import SimpleTokenizer, has_answer
from utils.util import str2bool
from utils.instruction import *
from utils.mapping import filter_and_map, ner, preprocess_dataset

def gpt_chat_completion(prompt, config: dict = {"top_p": 0.9, "max_tokens": 100}):
    client = OpenAI(timeout=10,max_retries=1)
    cnt = 0
    while True:
        try:
            response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            seed=42,
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            )
            break
        except:
            if cnt >= 3:
                return None
            cnt += 1
    return response.choices[0].message.content.strip()

def validate_by_answer_match(dataset: Dataset, ans_col_name: str, context_col_name: str, out_col_name: str) -> Dataset:
    result = []
    tokenizer = SimpleTokenizer()
    for row in dataset:
        answers, context = row[ans_col_name], row[context_col_name]
        if context is None:
            result.append(False)
            continue
        if has_answer(answers, context, tokenizer):
            result.append(False)
        else:
            result.append(True)
    dataset = dataset.add_column(out_col_name, result)
    return dataset

def validate_by_similarity(dataset: Dataset, query_col_name: str, key_col_name: str, out_col_name: str) -> Dataset:
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device="cuda")
    model.max_seq_length = 256
    result = []
    queries, keys = dataset[query_col_name], dataset[key_col_name]
    queries = [q if q is not None else "" for q in queries]
    keys = [k if k is not None else "" for k in keys]
    queries_encodings = model.encode(queries, batch_size=1000, show_progress_bar=True, normalize_embeddings=True)
    keys_encodings = model.encode(keys, batch_size=1000, show_progress_bar=True, normalize_embeddings=True)
    for i in range(len(queries_encodings)):
        if queries[i] == "" or keys[i] == "":
            result.append(False)
            continue
        if util.cos_sim(queries_encodings[i], keys_encodings[i]) >= 0.9:
            result.append(False)
        else:
            result.append(True)
    dataset = dataset.add_column(out_col_name, result)
    return dataset

def gen_ans_sent(dataset, args):
    questions, answers = dataset["question"], dataset["answers"]
    dataset = dataset.map(lambda x: {
        "gpt_answer_sentence": gpt_chat_completion(ANSWER_SENT.format(QUESTION=x["question"], ANSWER=x["answers"][0]))
        }, num_proc=min(16, os.cpu_count()))
    return dataset

def gen_adv_sent(dataset, args):
    dataset = ner(dataset, "gpt_answer_sentence", "entities")
    dataset = filter_and_map(dataset,
                             lambda x: x["entities_count"]>1,
                             lambda x: {"gpt_adv_sentence":gpt_chat_completion(ADV_SENT_PROMPT.format(ORIGINAL=x["gpt_answer_sentence"], REPLACE=x["entities"]))},
                             "gpt_adv_sentence")
    dataset = validate_by_answer_match(dataset, "answers", "gpt_adv_sentence", "valid_1")
    dataset = validate_by_similarity(dataset, "gpt_answer_sentence", "gpt_adv_sentence", "valid_2")
    dataset = dataset.map(lambda x: {"is_valid_sentence": x["valid_1"] and x["valid_2"]}, num_proc=os.cpu_count())
    dataset = dataset.remove_columns(["valid_1", "valid_2"])
    return dataset

def gen_adv_passage(dataset, args):
    dataset = filter_and_map(dataset,
                             lambda x: x["gpt_adv_sentence"] is not None,
                             lambda x: {"gpt_adv_passage":gpt_chat_completion(PASSAGE_PROMPT.format(CLAIM=x["gpt_adv_sentence"]), config={"top_p":0.9, "max_tokens":150})},
                             "gpt_adv_passage")
    dataset = validate_by_answer_match(dataset, "answers", "gpt_adv_passage", "is_valid_passage")
    return dataset

def gen_conflict_sent(dataset, args):
    pass
  
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate passages for claims")
    parser.add_argument("--test", type=str2bool, default=False)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--seed", type=int, help="Seed for reproducibility", default=42)
    parser.add_argument("--dataset", type=str, default="Atipico1/mrqa-test-mid")
    parser.add_argument("--tasks", type=str, nargs="+", default=[])
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="gpt3")
    parser.add_argument("--wandb", type=str2bool, default=True)
    args = parser.parse_args()
    dataset = load_dataset(args.dataset, split=args.split)
    if args.test:
        dataset = dataset.shuffle(seed=42).select(range(args.test_size))
    dataset = preprocess_dataset(dataset)
    if "answer_sent" in args.tasks:
        dataset = gen_ans_sent(dataset, args)
    if "adv_sent" in args.tasks:
        dataset = gen_adv_sent(dataset, args)
    if "adv_passage" in args.tasks:
        dataset = gen_adv_passage(dataset, args)
    if "conflict_sent" in args.tasks:
        dataset = gen_conflict_sent(dataset, args)
    if args.output_dir:
        _dataset = dataset.remove_columns(["entities", "entities_count"])
        _dataset.push_to_hub(args.output_dir)
    if args.wandb:
        import wandb
        dataset = dataset.remove_columns(["ctxs"])
        df = pd.DataFrame(dataset).sample(min(100, len(dataset)))
        df["answers"] = df["answers"].apply(lambda x: ", ".join(x))
        wandb.init(project="perturbation", name=args.run_name, config=vars(args))
        wandb.log({"samples":wandb.Table(dataframe=df)})