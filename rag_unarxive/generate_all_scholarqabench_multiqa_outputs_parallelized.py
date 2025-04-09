from dataclasses import dataclass
import os
import sys
import pprint
import subprocess
import signal
import time
import psutil
from psutil import Process
from typing import List

from check_server_health import check_server_health_request

LORA_MODEL_BASE_DIR = "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/"

BASE_MODELS = {
    #"Llama_8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama_3B": "meta-llama/Llama-3.2-3B-Instruct",
    #"Full_finetuned_10k_2epoch_bs64": "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_nolora_shuffled_split_10k_context_2_epoch_dist_train_bs64/checkpoint-3984",
    #"Full_finetuned_16k_1epoch_bs64": "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_nolora_shuffled_split_16k_context_1_epoch_dist_train_bs64/checkpoint-1992",
    #"Full_finetuned_16k_2epoch_bs64": "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_nolora_shuffled_split_16k_context_2_epoch_dist_train_bs64/checkpoint-3984",
    #"Full_finetuned_16k_5epoch_bs64": "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_nolora_shuffled_split_16k_context_5_epoch_dist_train_bs64/checkpoint-9965"
}

LORA_MODELS = {
    "Llama_8B": {},
    "Llama_3B": {
        #"finetuned_8k_1epoch": "model_checkpoints_100percent_lora32_shuffled_split/checkpoint-7970",
        #"finetuned_8k_2epoch": "model_checkpoints_100percent_lora32_shuffled_split_8k_context_2_epoch/checkpoint-3984",
        "finetuned_10k_2epoch": "model_checkpoints_100percent_lora32_shuffled_split_10k_context_2_epoch_dist_train_bs64/checkpoint-3984",
        #"finetuned_16k_1epoch": "model_checkpoints_100percent_lora32_shuffled_split_16k_context/checkpoint-3985",
        #"finetuned_16k_2epoch": "model_checkpoints_100percent_lora32_shuffled_split_16k_context_2_epoch_dist_train/checkpoint-1992",
        #"finetuned_16k_2epoch_bs16": "model_checkpoints_100percent_lora32_shuffled_split_16k_context_2_epoch_dist_train_bs16/checkpoint-15942",
        #"finetuned_16k_5epoch_bs64": "model_checkpoints_100percent_lora32_shuffled_split_16k_context_5_epoch_dist_train_bs64/checkpoint-9965"
    }
}

RAG_PORT = 8007
LLAMA_PORT = 8008

OUTPUT_BASE_DIR = "output_scholarqabench_multiqa_nobeam_stop_on_references"
LOG_BASE_DIR = OUTPUT_BASE_DIR + "_logs"

@dataclass
class RunSetup:
    name: str
    base_model: str
    lora_model: str|None
    use_rag: bool

RAG_ENABLED = [True]#, False]

runs: List[RunSetup] = []

#for name, model in BASE_MODELS.items():
#    for rag_en in RAG_ENABLED:
#        runs.append(RunSetup(name, model, "", rag_en))

for base_name, lora_models in LORA_MODELS.items():
    for lora_name, lora_dir in lora_models.items():
        runs.append(RunSetup(f"{base_name}_{lora_name}", BASE_MODELS[base_name], os.path.join(LORA_MODEL_BASE_DIR, lora_dir), True))

pprint.pp(runs)

if not os.path.exists(OUTPUT_BASE_DIR):
    os.makedirs(OUTPUT_BASE_DIR)

if not os.path.exists(LOG_BASE_DIR):
    os.makedirs(LOG_BASE_DIR)




for i,run in enumerate(reversed(runs)):
    print(f"Generation run {i}/{len(runs)}: {run}")
    env = {}
    env.update(os.environ)
    env["RUN_NAME"] = run.name
    env["BASE_MODEL"] = run.base_model
    env["LORA_MODEL"] = run.lora_model
    if run.use_rag:
        env["USE_RAG"] = "True"
    env["OUTPUT_BASE_DIR"] = OUTPUT_BASE_DIR
    env["RAG_PORT"] = str(RAG_PORT + 2*i)
    env["LLAMA_PORT"] = str(LLAMA_PORT + 2*i)

    print("\n\n" + 20*"=" + "\n" + f"{run}" + "\n")

    name = run.name
    if not run.use_rag:
        name += "_NO_RAG"
    log_file = os.path.join(LOG_BASE_DIR, f"{name}.log")
    env["LOG_BASE_FILE"] = log_file
    
    sbatch_process = subprocess.Popen(["bash", "-c", f"source ~/.bashrc; set_capella_env; sbatch -o {log_file} -e {log_file} --job-name=gen_sqa_mult_{name} generate_scholarqabench_multiqa_outputs_parallelized_runner.sbatch"], env=env)
    
    sbatch_process.wait()

    time.sleep(1)