from datasets import load_dataset, Dataset, DatasetDict
from trl import SFTConfig, SFTTrainer
from peft import LoraModel, LoraConfig

use_lora = True

def change_sparql_dict_to_str(dataset):
    dataset["prompt"] = dataset["prompt"]["string"]
    dataset["completion"] = dataset["completion"]["sparql"]
    return dataset

print('load awalesushil/DBLP-QuAD')
dataset = load_dataset("awalesushil/DBLP-QuAD", split="train")

# format dataset for SFTTrainer
print('change dataset for SFTTrainer')
for del_key in ['id', 'query_type', 'paraphrased_question', 'template_id', 'entities', 'temporal', 'held_out', 'relations']:
    dataset = dataset.remove_columns(del_key)

dataset = dataset.rename_column("question", "prompt")
dataset = dataset.rename_column("query", "completion")
dataset = dataset.map(change_sparql_dict_to_str)

train_val_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_val_split['train']
eval_dataset = train_val_split['test']


training_args = SFTConfig(
    max_seq_length=512,
    output_dir="./Llama-3.2-3B-SPARQL-SFT-weight_decay_warmup",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    lr_scheduler_type="linear", #cosine
    bf16=False,
    gradient_accumulation_steps=1, #16
    logging_steps=500, #50
    eval_steps=250, #None
    weight_decay=0.01, #0
    warmup_steps=200, #0,
    max_grad_norm=1.0, #0.01,
    num_train_epochs=3, #1,
    load_best_model_at_end=False, #True,
    eval_strategy="steps", #"no",
)

if use_lora:
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        modules_to_save=["lm_head", "embed_token"],
        task_type="CAUSAL_LM",
        target_modules="all-linear",
        #target_modules=[
        #    "gate_proj",
        #    "down_proj",
        #    "up_proj",
        #    "q_proj",
        #    "v_proj",
        #    "k_proj",
        #    "o_proj",
        #],
    )

    trainer = SFTTrainer(
        "meta-llama/Llama-3.2-3B-Instruct",
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        peft_config=peft_config
    )

else:
    print('setup SFTTrainer')
    trainer = SFTTrainer(
        "meta-llama/Llama-3.2-3B-Instruct",
        train_dataset=dataset,
        args=training_args,
    )


print('finetune')
trainer.train()


"""
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
    r=8,
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


training_args = SFTConfig(
    max_seq_length=8192,
    output_dir="./model_checkpoints_100percent_second_try/",
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
"""
