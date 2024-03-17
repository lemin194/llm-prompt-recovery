import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig
import re
from huggingface_hub import login
login(token='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')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL_PATH = 'google/gemma-7b-it'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def chat(prompt, max_new_token=256, **generate_args):

  messages = [
      {
          "role": "user",
          "content": prompt + '\n ',
      }
  ]
  encoded_input = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to('cuda')
  # print(encoded_input.shape)
  # print(torch.max(encoded_input), tokenizer.vocab_size)
  with torch.no_grad():
      encoded_output = model.generate(encoded_input, max_new_tokens=max_new_token, do_sample=True, pad_token_id=tokenizer.eos_token_id,
                                      **generate_args)
  
  decoded_output = tokenizer.batch_decode(encoded_output, skip_special_tokens=True)[0]
  decoded_output = re.sub(r"[\s\S]*model\n", '', decoded_output, 1).replace(prompt, '')
  
  return decoded_output


def batch_chat(prompts, max_new_token=256, **generate_args):
  messages = []
  for prompt in prompts:
      messages.append([{
          "role": "user",
          "content": prompt + '\n ',
      }])

  chat_messages = [tokenizer.apply_chat_template(message, return_tensors="pt", add_generation_prompt=True, tokenize=False) for message in messages]
  encoded_input = tokenizer(chat_messages, padding=True, return_tensors='pt').to('cuda')
  with torch.no_grad():
      encoded_output = model.generate(**encoded_input, max_new_tokens=max_new_token, do_sample=True, pad_token_id=tokenizer.eos_token_id,
                                      **generate_args)

  decoded_outputs = tokenizer.batch_decode(encoded_output, skip_special_tokens=True)
  decoded_outputs = [re.sub(r"[\s\S]*model\n", '', decoded_output, 1).replace(prompt, '').strip() for prompt, decoded_output in zip(prompts, decoded_outputs)]

  try: del encoded_input, encoded_output, messages, chat_messages
  except: pass
  return decoded_outputs

torch.cuda.empty_cache()
import gc
gc.collect(2)
try: del model
except: pass

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, 
    # bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                              quantization_config=quantization_config, 
                                              torch_dtype=torch.bfloat16,
                                              attn_implementation='flash_attention_2',
                                            )

def rewrite_text(original_text, rewrite_prompt):
  prompt = f"""
{rewrite_prompt}
\"""{original_text}\"""
"""
  res = chat(prompt)
  return res.strip()

def batch_rewrite_text(original_texts, rewrite_prompts):
  prompts = []
  for i in range(len(original_texts)):
    original_text, rewrite_prompt = original_texts[i], rewrite_prompts[i]
    prompt = f"""
  {rewrite_prompt}
  \"""{original_text}\"""
  """
    prompts.append(prompt)
  
  res = batch_chat(prompts)
  return res

import os
import pandas as pd
from tqdm import tqdm

filenames = [os.path.join('./input/train', f) for f in os.listdir('./input/train') if f.startswith('part') and f.endswith('.csv')]
filenames = ['./input/train/part1.csv']
ckpt_steps = 50
batch_size = 4
for filename in filenames:
  print(filename)
  part = pd.read_csv(filename)
  part.rename(columns={'rewrite_prompts': 'rewrite_prompt'}, errors='ignore', inplace=True)
  if 'rewritten_text' not in part.columns:
    part['rewritten_text'] = 'none'
  for i in tqdm(range(0,len(part), batch_size)):
    torch.cuda.empty_cache()
    import gc
    gc.collect(2)
    if 'none' not in list(part.loc[i : i + batch_size - 1, 'rewritten_text']):
      # print(part.loc[i : i + batch_size - 1, 'rewritten_text'])
      continue
    part.loc[i:i+batch_size - 1, 'rewritten_text'] = batch_rewrite_text(
      list(part.loc[i:i+batch_size - 1, 'original_text']), list(part.loc[i:i+batch_size - 1, 'rewrite_prompt']))
    
    # try:
    #   part.loc[i:i+batch_size - 1, 'rewritten_text'] = batch_rewrite_text(
    #     list(part.loc[i:i+batch_size - 1, 'original_text']), list(part.loc[i:i+batch_size - 1, 'rewrite_prompt']))
    # except Exception as e:
    #   part.loc[i:i+batch_size - 1, 'rewritten_text'] = 'none'
    #   print("Error at %s, i=%d\n%s" % (filename, i, str(e)))
    part.to_csv(filename, index=False)
  
  part.to_csv(filename, index=False)
  
print('done!')