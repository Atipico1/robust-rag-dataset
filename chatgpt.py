from openai import OpenAI
import pandas as pd
from tqdm.auto import tqdm
import argparse, time
from datasets import load_dataset
from src.utils import str2bool

client = OpenAI(timeout=10,max_retries=1)
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
