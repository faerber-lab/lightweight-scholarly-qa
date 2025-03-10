from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import torch
import os
import shutil

from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset


from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader


training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    bfloat16=True
)
#from unsloth import FastLanguageModel

# Base model
model_base_name = "Llama-3.2-3B-Instruct"
model_name = f"meta-llama/{model_base_name}"   
    
model_in_cache_name = f"models--{model_name.split('/')[0]}--{model_name.split('/')[1]}"

model_snapshot = "0e9e39f249a16976918f6564b8830bc894c89659" if model_name == "meta-llama/Llama-3.1-8B-Instruct" else "0cb88a4f764b7a12671c53f0838cd831a0843b95"
    
cache_path = os.path.join(os.path.expanduser('~/.cache/huggingface/hub/'), model_in_cache_name, f"snapshots/{model_snapshot}")
tmp_base_path = "/tmp/s9650707/models/"
tmp_path = os.path.join(tmp_base_path, model_in_cache_name)
if os.path.exists(cache_path):
    print("Model exists in cache -> copy to tmp if necessary and use")
    if not os.path.exists(tmp_base_path):
        os.makedirs(tmp_base_path)
    if not os.path.exists(tmp_path):
        print("Copying model to tmp")
        shutil.copytree(cache_path, tmp_path)
    model_path = tmp_path
else:
    print("Model does not exist in cache -> download")
    model_path = model_name

base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16) #"Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(model_path) #"Qwen/Qwen2-0.5B-Instruct")  # Changed to load from fine-tuned path
tokenizer.pad_token=tokenizer.eos_token


# Peft model
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
peft_model = get_peft_model(base_model, peft_config)

peft_model.num_parameters()
peft_model.print_trainable_parameters()


model=peft_model

target_model_path = f"./finetuned/{model_name}-lora"""
"""
training_args = TrainingArguments(
    output_dir=f"{target_model_path}/",
    learning_rate=5e-4,
    per_device_train_batch_size=4,
    #per_device_eval_batch_size=2,
    num_train_epochs=10,
    #eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    gradient_accumulation_steps=16,
    logging_steps=500,
    weight_decay=0.01,
    warmup_steps=500,
)"""

dataset = load_dataset("OpenScholar/OS_Train_Data", split="train[:1%]", cache_dir="/tmp/s9650707/datasets")

dataloader = DataLoader(dataset, batch_size=training_args.per_device_train_batch_size)

#for j in range(100):
#    msg = dataset[j]['messages']
#    print([f"{msg[i]['role']}, {len(tokenizer(msg[i]['content'])['input_ids'])}" for i in range(len(msg))])

#context_length=4096
#response_template = "<|start_header_id|>assistant<|end_header_id|>"
#dataset = tokenizer(dataset, add_special_tokens=True, truncation=True)['messages']#, max_length=context_length)["input_ids"]
#data_collator =  DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response_template, mlm=False)

#x = data_collator

model.train()

training_args = SFTConfig(
    max_seq_length=7000,
    output_dir="/tmp/s9650707/models/",
    per_device_train_batch_size=2,
    learning_rate=1e-6,
    #fp16=True,
    gradient_accumulation_steps=8,
    logging_steps=5,
    weight_decay=0.01,
    warmup_steps=5,
)
trainer = SFTTrainer(
    model=model,
    #tokenizer=tokenizer,
    train_dataset=dataset,
    eval_dataset=dataset,
    #data_collator=data_collator,
    args=training_args,
)
#with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
#with torch.autograd.detect_anomaly():
trainer.train()