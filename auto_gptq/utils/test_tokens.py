from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from functools import partial

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=True)

wikidata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
wikilist = [' \n' if s == '' else s for s in wikidata['text']]

def method1():
    # Tokenize the full text in one pass
    text = ''.join(wikilist)

    method1_tokens = tokenizer(text, truncation=False, add_special_tokens=False, return_tensors='pt').input_ids

    return method1_tokens[0]

def method1b():
    # Iterate through each sample of the dataset, tokenizing one by one. Then concat at the end.
    tokens = []
    for sample in wikilist:
        output = tokenizer(sample, truncation=False, add_special_tokens=False, return_tensors='pt').input_ids
        tokens.append(output)

    method1b_tokens = torch.LongTensor()
    for item in tokens: # Concat tokenized samples into one tensor
        input_ids = item[0]
        method1b_tokens = torch.cat((method1b_tokens, input_ids), dim=0)
    return method1b_tokens

def method2():
    data_name_or_path="wikitext"
    prompt_col_name="text"
    load_fn_kwargs = { "name": 'wikitext-2-raw-v1', "split": "test"}

    ds = load_dataset(data_name_or_path, **load_fn_kwargs)

    def make_data_block(samples,
                        prompt_col_name,
                        tokenizer,
                        preprocess_fn,
                        add_special_tokens):
        if preprocess_fn:
            samples = preprocess_fn(samples) 
            
        prompts = samples[prompt_col_name]

        tokenized_prompts = tokenizer(prompts, truncation=False, add_special_tokens=add_special_tokens,return_tensors='pt').input_ids
        
        new_samples = { "input_ids": [] }
        new_samples["input_ids"].append(tokenized_prompts)
        
        return new_samples
        
    def preprocess_fn(samples):
        return {prompt_col_name: ' \n' if samples[prompt_col_name] == '' else samples[prompt_col_name]}

    ds = ds.map(make_data_block,
                fn_kwargs={
                    "prompt_col_name": prompt_col_name,
                    "tokenizer": tokenizer,
                    "preprocess_fn": callable(preprocess_fn) and partial(preprocess_fn) or None,
                    "add_special_tokens": False
                    }
                )
    dl = DataLoader(ds)
        
    method2_tokens = torch.LongTensor()  # initialize an empty tensor
    for item in dl:
        input_ids = torch.stack(item['input_ids'][0][0]).reshape(1, -1)
        method2_tokens = torch.cat((method2_tokens, input_ids), dim=1)  # concatenate the new tenso
        
    return method2_tokens[0]

m1b_tokens = method1b()
m1_tokens = method1()
m2_tokens = method2()

print("method 1 token len:", len(m1_tokens))
print("method 1b token len:", len(m1b_tokens))
print("method 2 token len:", len(m2_tokens))
print("m1 tokens:", end='')
for i, m1 in enumerate(m1_tokens):
    if i < 20:
        print(f"{m1}, ", end='')

print("\n\n\n")

print("m1b tokens:", end='')
for i, m1b in enumerate(m1b_tokens):
    if i < 20:
        print(f"{m1b}, ", end='')
print("\n")

print("m2 tokens:", end='')
for i, m2 in enumerate(m2_tokens):
    if i < 20:
        print(f"{m2}, ", end='')
print("\n")