from openai import OpenAI
import pandas as pd
from tqdm.auto import tqdm
import argparse, time
from datasets import load_dataset
from src.utils import str2bool
from src.prompt import *

client = OpenAI(timeout=10,max_retries=1)

def gpt_chat_completion(prompt, config: dict = {"top_p": 0.9, "max_tokens": 100}):
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
    return response.choices[0].message.content.strip()

def gen_adv_sent(dataset, args):
    answer_sents, entities = dataset["clear_answer_sent"], dataset["entities"]
    output = []
    for i in tqdm(range(len(answer_sents)), desc="Adv Sent Generating"):
        if entities[i] == []:
            output.append(None)
        else:
            prompt = ADV_SENT_PROMPT.format(ORIGINAL=answer_sents[i], REPLACE=entities[i])
            response = gpt_chat_completion(prompt)
            output.append(response)
    dataset = dataset.add_column("gpt_adv_sent", output)
    dataset.push_to_hub("Atipico1/mrqa-adv-test-adv-gpt-passage-entity")

def gen_adv_passage(dataset, args):
    dataset = dataset.map(lambda x: {"adv_passage": gpt_chat_completion(
        PASSAGE_PROMPT.format(CLAIM=x["adv_sent"]), config={"top_p":0.9, "max_tokens":150}) if x["adv_sent"] else None})
    dataset.push_to_hub(args.output_dir)

def gen_new_sent(dataset, args):
    import spacy
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    docs = list(nlp.pipe(dataset["context"]))
    ents, ents_count = [],[]
    for doc in docs:
        if doc.ents:
            ents.append(", ".join(['"'+ent.text+'"' for ent in doc.ents]))
            ents_count.append(len(doc.ents))
        else:
            ents.append(None)
            ents_count.append(0)
    dataset = dataset.add_column("entities", ents)
    dataset = dataset.add_column("entities_count", ents_count)
    dataset = dataset.map(lambda x: {"adv_sent":gpt_chat_completion(
        NEW_SENT.format(sentence=x["context"],entities=x["entities"])) if x["entities_count"]>1 else None})
    if args.test:
        [print(x) for x in dataset["adv_sent"]]
    else:
        dataset.push_to_hub("Atipico1/nq-test-adv_sent")
  
if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Generate passages for claims")
    parser.add_argument("--test", type=str2bool, help="Generate passages for the beginning of the claim", default=False)
    parser.add_argument("--seed", type=int, help="Seed for reproducibility", default=42)
    parser.add_argument("--dataset", type=str, default="Atipico1/mrqa-test-mid")
    parser.add_argument("--task", type=str, default="adv_passage")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()
    cnt = 0
    while True and cnt<10:
        try:
            dataset = load_dataset(args.dataset, split=args.split)
            break
        except:
            cnt += 1
            time.sleep(5)
            print(f"Dataset not found {cnt} times")
    if args.test:
        dataset = dataset.select(range(10))
    if args.task == "adv_sent":
        gen_adv_sent(dataset, args)
    elif args.task == "adv_passage":
        gen_adv_passage(dataset, args)
    elif args.task == "new_sent":
        gen_new_sent(dataset, args)
