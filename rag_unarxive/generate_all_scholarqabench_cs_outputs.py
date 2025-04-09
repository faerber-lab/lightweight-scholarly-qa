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
    "Llama_8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Llama_3B": "meta-llama/Llama-3.2-3B-Instruct",
    "Full_finetuned_16k_2epoch_bs64": "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_nolora_shuffled_split_16k_context_2_epoch_dist_train_bs16/checkpoint-3984"
}

LORA_MODELS = {
    "Llama_8B": {},
    "Llama_3B": {
        #"finetuned_8k_1epoch": "model_checkpoints_100percent_lora32_shuffled_split/checkpoint-7970",
        #"finetuned_8k_2epoch": "model_checkpoints_100percent_lora32_shuffled_split_8k_context_2_epoch/checkpoint-3984",
        #"finetuned_16k_1epoch": "model_checkpoints_100percent_lora32_shuffled_split_16k_context/checkpoint-3985",
        "finetuned_16k_2epoch": "model_checkpoints_100percent_lora32_shuffled_split_16k_context_2_epoch_dist_train/checkpoint-1992",
        #"finetuned_16k_2epoch_bs16": "model_checkpoints_100percent_lora32_shuffled_split_16k_context_2_epoch_dist_train_bs16/checkpoint-15942"
    }
}

RAG_PORT = 8003
LLAMA_PORT = 8004

OUTPUT_BASE_DIR = "output_scholarqabench_cs/"

@dataclass
class RunSetup:
    name: str
    base_model: str
    lora_model: str|None
    use_rag: bool

RAG_ENABLED = [False, True]

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


llama_process = None
rag_process = None
generate_process = None

def stop_process(process, kill_timeout=2):
    print("stopping process...")
    for child in Process(process.pid).children(recursive=True):
        try:
            child.terminate()
            child.wait(timeout=kill_timeout)
        except psutil.TimeoutExpired:
            child.kill()
        except:
            pass
    try:
        process.terminate()
        process.wait(timeout=kill_timeout)
    except psutil.TimeoutExpired:
        process.kill()
    except:
        pass

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    if llama_process:
        try:
            stop_process(llama_process)
        except:
            pass
    if rag_process:
        try:
            stop_process(rag_process)
        except:
            pass
    if generate_process:
        try:
            stop_process(generate_process)
        except:
            pass
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

rag_process = subprocess.Popen(["bash", "-c", "source ~/.bashrc; set_capella_env && . ./start_rag_server.sh"])


for i,run in enumerate(runs):
    print(f"Generation run {i}/{len(runs)}: {run}")
    env = {}
    env.update(os.environ)
    env["RUN_NAME"] = run.name
    env["BASE_MODEL"] = run.base_model
    env["LORA_MODEL"] = run.lora_model
    if run.use_rag:
        env["USE_RAG"] = "True"

    print("\n\n" + 20*"=" + "\n" + f"{run}" + "\n")

    llama_process = subprocess.Popen(["bash", "-c", "source ~/.bashrc; set_scholarqabench_env && . ./start_llama_server_vllm.sh"], env=env)


    while not check_server_health_request(port=RAG_PORT):
        print("Waiting for RAG server to be loaded!")
        time.sleep(2)

    while not check_server_health_request(port=LLAMA_PORT):
        print("Waiting for Llama server to be loaded!")
        time.sleep(2)
    
    name = run.name
    if not run.use_rag:
        name += "_NO_RAG"
    generate_output_dir = os.path.join(OUTPUT_BASE_DIR, name)
    print(f"{name=}, {generate_output_dir=}")
    generate_process = subprocess.Popen(["bash", "-c", f"source ~/.bashrc; set_capella_env && python generate_scholarqabench_cs_output.py --output_dir {generate_output_dir}{'' if run.use_rag else ' --no_rag'}"])
    generate_process.wait()

    stop_process(llama_process)
    print("Waiting for Llama to stop")
    llama_process.wait()


stop_process(rag_process)