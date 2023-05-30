from transformers import AutoTokenizer
from datasets import load_dataset
import torch

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=True)

wikidata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
wikilist = [' \n' if s == '' else s for s in wikidata['text']]

def tokenize_one_pass():
    # Tokenize the full text in one pass
    text = ''.join(wikilist)

    method1_tokens = tokenizer(text, truncation=False, add_special_tokens=False, return_tensors='pt').input_ids

    return method1_tokens[0]

def tokenize_by_sample():
    # Iterate through each sample of the dataset, tokenizing one by one. Then concat at the end.
    tokens = []
    count=0
    for sample in wikilist:
        output = tokenizer(sample, truncation=False, add_special_tokens=False, return_tensors='pt').input_ids
        if count < 10:
            print(f"||{sample}||")
            print(output)
        tokens.append(output)
        count = count + 1

    method2_tokens = torch.LongTensor()
    for item in tokens: # Concat tokenized samples into one tensor
        input_ids = item[0]
        method2_tokens = torch.cat((method2_tokens, input_ids), dim=0)
    return method2_tokens

m1_tokens = tokenize_one_pass()
m2_tokens = tokenize_by_sample()

print("method 1 (tokenize in one pass) token len:", len(m1_tokens))
print("method 2 (tokenize sample-by-sample) token len:", len(m2_tokens))
print("m1 tokens:", end='')
for i, m1 in enumerate(m1_tokens):
    if i < 20:
        print(f"{m1}, ", end='')

print("\n\n\n")

print("m2 tokens:", end='')
for i, m2 in enumerate(m2_tokens):
    if i < 20:
        print(f"{m2}, ", end='')
print("\n")