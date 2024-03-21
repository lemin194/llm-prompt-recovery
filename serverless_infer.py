import requests
import os

API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"
hf_token = os.environ['HF_TOKEN']

from huggingface_hub import InferenceClient
import os

API_URL = "https://api-inference.huggingface.co/models/google/gemma-7b-it"
hf_token = os.environ['HF_TOKEN']

client = InferenceClient(API_URL, hf_token)
def chat(prompt, **gen_kwargs):
  gen_kwargs.setdefault('max_new_tokens', 256,)
  gen_kwargs.setdefault('top_k', 20,)
  gen_kwargs.setdefault('stop_sequences', ["\nUser:", "<|endoftext|>", "</s>"],)
  res = client.text_generation(prompt, **gen_kwargs)
  # res = ""
  # for r in stream:
  #   # skip special tokens
  #   if r.token.special:
  #       continue
  #   # stop if we encounter a stop sequence
  #   if r.token.text in gen_kwargs["stop_sequences"]:
  #       break
  #   # yield the generated token
  #   res += r.token.text
  return res
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b-it')
def gen_prompt(example):
  return f"""Instruction: {example['rewrite_prompt']}
\"""{example['original_text']}\"""
"""
import pandas as pd
from tqdm import tqdm
import time

fnames = ['input/train/part3.csv']
ckpt_steps = 50
wait_interval = 60
for fname in fnames:
  print(fname)
  df = pd.read_csv(fname)
  for i in tqdm(range(len(df))):
    row = df.iloc[i]
    if row.rewritten_text != 'none': continue
    prompt = gen_prompt(row)
    res = 'none'
    while res == 'none':
      try: res = chat(prompt)
      except Exception as e:
        print(e)
        df.to_csv(fname, index=False)
        time.sleep(wait_interval)
        res = 'none'
    # try: res = chat(prompt)
    # except: res = 'none'
    df.loc[i, 'rewritten_text'] = res
    
    # print(df.loc['rewritten_text', i])
    # break
    if (i + 1) % ckpt_steps == 0:
      df.to_csv(fname, index=False)
      # print('saved')