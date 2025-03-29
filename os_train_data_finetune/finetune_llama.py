print("loading transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("loading torch")
import torch

print("loading others")
import os
import shutil

from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import load_dataset

print("done loading libs")
print("loading model")

# from unsloth import FastLanguageModel

# Base model
model_base_name = "Llama-3.2-3B-Instruct"
model_name = f"meta-llama/{model_base_name}"

model_in_cache_name = f"models--{model_name.split('/')[0]}--{model_name.split('/')[1]}"

model_snapshot = (
    "0e9e39f249a16976918f6564b8830bc894c89659"
    if model_name == "meta-llama/Llama-3.1-8B-Instruct"
    else "0cb88a4f764b7a12671c53f0838cd831a0843b95"
)

cache_path = os.path.join(
    os.path.expanduser("~/.cache/huggingface/hub/"),
    model_in_cache_name,
    f"snapshots/{model_snapshot}",
)
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


base_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
)  # "Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained(
    model_path
)  # "Qwen/Qwen2-0.5B-Instruct")  # Changed to load from fine-tuned path
tokenizer.pad_token = tokenizer.eos_token


# Peft model
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[
        "gate_proj",
        "down_proj",
        "up_proj",
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
)
peft_model = get_peft_model(base_model, peft_config)

peft_model.num_parameters()
peft_model.print_trainable_parameters()


model = peft_model

print("loading dataset")

dataset = load_dataset(
    "OpenScholar/OS_Train_Data",
    num_proc=4,
    cache_dir="/tmp/s9650707/datasets",
)

"""train_dataset = load_dataset(
    "OpenScholar/OS_Train_Data",
    num_proc=4,
    split="train[:98%]",
    cache_dir="/tmp/s9650707/datasets",
)

eval_dataset = load_dataset(
    "OpenScholar/OS_Train_Data",
    num_proc=4,
    split="train[-2%:]",
    cache_dir="/tmp/s9650707/datasets",
)"""

split_dataset = dataset["train"].train_test_split(test_size=0.02)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]


print("size of train dataset:", len(train_dataset))
print("size of eval dataset:", len(eval_dataset))

all_datasets = {}
examples = {}
for i,data in enumerate(dataset['train']):
    ds = data['dataset']
    all_datasets[ds] = all_datasets[ds]+1 if ds in all_datasets else 1
    if ds not in examples:
        examples[ds] = (data['id'], data['messages'][0]['content'], data['messages'][1]['content'])
        #print("train", i, data['dataset'], data['id'], data['messages'][0]['content'][:250])


train_datasets = {}
for i,data in enumerate(train_dataset):
    ds = data['dataset']
    train_datasets[ds] = train_datasets[ds]+1 if ds in train_datasets else 1
    
eval_datasets = {}
for i,data in enumerate(eval_dataset):
    ds = data['dataset']
    eval_datasets[ds] = eval_datasets[ds]+1 if ds in eval_datasets else 1


print(f"{train_datasets=}\n{eval_datasets=}\n{all_datasets=}")

with open("examples.txt", "w") as f:
    for name, example in sorted(examples.items()):
        f.write(f"### {name} (id {example[0]})\n##### Request\n{example[1]}\n##### Reply\n{example[2]}\n\n")


exit()

model.train()

training_args = SFTConfig(
    max_seq_length=8192,
    output_dir="./model_checkpoints_100percent_lora32_shuffled_split/",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    learning_rate=0.000005,
    lr_scheduler_type="cosine",
    bf16=True,
    gradient_accumulation_steps=16,
    logging_steps=50,
    eval_steps=250,
    weight_decay=0.01,
    warmup_steps=200,
    max_grad_norm=0.01,
    num_train_epochs=1,
    load_best_model_at_end=True,
    eval_strategy="steps",
)
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
)

trainer.train()
