from accelerate.utils import BnbQuantizationConfig
from accelerate import Accelerator, notebook_launcher
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, \
                        get_cosine_schedule_with_warmup, set_seed
import transformers
import optimum

from datasets import load_dataset,Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, \
                        get_cosine_schedule_with_warmup, set_seed
from peft import LoraConfig, TaskType, get_peft_model
from accelerate import Accelerator, notebook_launcher
from accelerate.utils import set_seed
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from tqdm.notebook import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
import glob
from collections import OrderedDict
import re

import os

import pandas as pd
from sklearn.model_selection import train_test_split


# MODEL_PATH = "distilbert/distilgpt2"
# MODEL_PATH = "gpt2"
# MODEL_PATH = "microsoft/deberta-v3-base"
# MODEL_PATH = "distilbert/distilroberta-base"
MODEL_PATH = 'mistralai/Mistral-7B-Instruct-v0.2'
# MODEL_PATH = "google/gemma-7b-it"

max_length = 1024

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.model_max_length = max(tokenizer.model_max_length, max_length)

def create_data(df, tokenizer, split='train'):
  data = Dataset.from_pandas(df[[
    # 'reverse_prompt',
    'rewrite_prompt', 'original_text', 'rewritten_text'
  ]],split=split)
  
  data = data.map(lambda samples: tokenizer(samples["original_text"], max_length=max_length, truncation=True), batched=True)
  data = data.map(lambda samples: tokenizer(samples["rewritten_text"], max_length=max_length, truncation=True), batched=True)
  data = data.map(lambda samples: tokenizer(samples["rewrite_prompt"], max_length=max_length, truncation=True), batched=True)
  
  return data

train, test = (
  create_data(pd.read_csv('./input/mydata/train.csv'), tokenizer, 'train'),
  create_data(pd.read_csv('./input/mydata/test.csv'), tokenizer, 'test'),
)
print('(train, test) =', len(train), len(test))


def truncate_txt(text, length):
    text_list = text.split()
    
    if len(text_list) <= length:
        return text
    
    return " ".join(text_list[:length])


def gen_prompt(og_text, rewritten_text, truncate_length=200):
    
    # Truncate the texts to first 200 words for now
    # As we are having memory issues on Mixtral8x7b
    og_text = truncate_txt(og_text, truncate_length)
    rewritten_text = truncate_txt(rewritten_text, truncate_length)
    
    return f"""
You are given 2 essays, the Rewritten essay was created from the Original essay using the google Gemma model.
Analyzing the changes in style, theme, etc., please come up with a prompt that must have been used to guide the transformation from the original to the rewritten essay.
Start directly with the prompt, output should be one line only.

Original Essay:
\"""{og_text}\"""

Rewritten Essay:
\"""{rewritten_text}\"""

""".strip()


def formatting_func(example):
  output_texts = []
  for i in range(len(example['original_text'])):
    prompt = tokenizer.apply_chat_template(
      [
        {
          'role': 'user',
          'content' : gen_prompt(example['original_text'][i], example['rewritten_text'][i]),
        }, {
          'role': 'assistant',
          'content' : example['rewrite_prompt'][i],
        },
      ],
      tokenize=False,
    )
    text = f"{prompt}{tokenizer.eos_token}"
    output_texts.append(text)
  return output_texts


from trl import SFTTrainer

def main(batch_size: int, num_epochs: int, lr: float, grad_accumulation_steps: int, 
         checkpointing_steps: int, save_path: str, ckpt_path: str,
         num_warmup_steps: int=0, r: int=4, lora_alpha: int=32, lora_dropout: float=0.1,
         eval_steps=None):
    set_seed(1234)
    
    accelerator = Accelerator(gradient_accumulation_steps=grad_accumulation_steps)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(MODEL_PATH)
    # Load checkpoint
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                                 quantization_config=quantization_config, 
                                                 torch_dtype=torch.bfloat16)
    
    
    accelerator.print(model)
    
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, 
                             inference_mode=False, r=r, 
                             lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             target_modules=
                            lora_target_modules_dict.get(MODEL_PATH,
                              lora_target_modules_dict['mistralai/Mistral-7B-Instruct-v0.2'],)
                            )
    peft_model = get_peft_model(model, peft_config)
    
    if accelerator.is_local_main_process:
        peft_model.print_trainable_parameters()
    
    use_tf32 = True
    if use_tf32:
      torch.backends.cuda.matmul.allow_tf32 = True
      torch.backends.cudnn.allow_tf32 = True
    
    print(list(model.parameters())[0][0, 0])
    trainer = SFTTrainer(
        model=model,
        train_dataset=train,
        eval_dataset=test,
        args=transformers.TrainingArguments(
            output_dir=save_path,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=grad_accumulation_steps,
            gradient_checkpointing=True,
            warmup_steps=2,
            max_steps=num_epochs,
            # num_train_epochs=num_epochs,
            load_best_model_at_end=True,
            evaluation_strategy='steps',
            save_strategy='steps',
            eval_steps=eval_steps or checkpointing_steps,
            save_steps=checkpointing_steps,
            learning_rate=lr,
            fp16=True,
            tf32=use_tf32,
            logging_steps=1,
            optim="paged_adamw_8bit",
            logging_dir='./logs/',
            # save_total_limit=3,
            save_only_model=True,
        ),
        peft_config=peft_config,
        formatting_func=formatting_func,
        # data_collator=collator,
    )
    resume_from_checkpoint = True
    from transformers.trainer_utils import get_last_checkpoint
    if get_last_checkpoint(save_path) is None: resume_from_checkpoint = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint,)
    
    trainer.save_model(save_path)
    
    print(list(model.parameters())[0][0, 0])
    # model = get_peft_model(model, peft_config)
    # model = model.merge_and_unload()
    # model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)
    
lora_target_modules_dict = {
  'gpt2': ['c_attn'],
  'distilbert/distilgpt2': ['c_attn'],
  'distilbert/distilroberta-base': ['query', 'key', 'value'],
  'mistralai/Mistral-7B-Instruct-v0.2': ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
  'google/gemma-7b-it': ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
}
import json
os.makedirs('./settings', exist_ok=True)
json.dump(lora_target_modules_dict, open('./settings/lora_target_modules.json', 'w'))

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
print(tokenizer.pad_token_id)




batch_size = 1
grad_accumulation_steps = 4
num_epochs = 10
lr = 5e-5
checkpointing_steps = 10
eval_steps = 10
save_path = os.path.join('./working/trained_models/', MODEL_PATH)
r = 32
lora_alpha = 32
lora_dropout = 0.05

# If ckpt_path is a real path (os.path.isfile(ckpt_path) is True),
# then the checkpoint will be loaded
ckpt_path = os.path.join(save_path, 'checkpoint.pth')
print(ckpt_path)
kwargs = {
'batch_size':batch_size, 'num_epochs':num_epochs, 'lr':lr, 'grad_accumulation_steps':grad_accumulation_steps, 
'checkpointing_steps':checkpointing_steps, 'save_path':save_path, 'ckpt_path':ckpt_path, 'r':r, 'lora_alpha':lora_alpha, 'lora_dropout':lora_dropout,
'eval_steps': eval_steps,
}

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)
# notebook_launcher(main, args, num_processes=1)
main(**kwargs)