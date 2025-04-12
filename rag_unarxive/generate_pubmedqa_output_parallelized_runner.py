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

RAG_PORT = int(os.environ["RAG_PORT"])
LLAMA_PORT = int(os.environ["LLAMA_PORT"])

RUN_NAME = os.environ["RUN_NAME"]
BASE_MODEL = os.environ["BASE_MODEL"]
LORA_MODEL = os.environ.get("LORA_MODEL", None)
USE_RAG = True if os.environ.get("USE_RAG", "False") == "True" else False

USE_ORIG_DS = True if os.environ.get("USE_ORIG_DS", "False") == "True" else False
TEST_SET_ONLY = True if os.environ.get("TEST_SET_ONLY", "False") == "True" else False

OUTPUT_BASE_DIR = os.environ["OUTPUT_BASE_DIR"]
LOG_BASE_FILE = os.environ["LOG_BASE_FILE"]

print("\n\n" + 20*"=" + "\n" + f"{RUN_NAME=} {BASE_MODEL=}, {LORA_MODEL=}, {USE_RAG=}, {USE_ORIG_DS=}, {RAG_PORT=}, {LLAMA_PORT=}" + "\n")


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

rag_log = LOG_BASE_FILE + ".rag"
with open(rag_log, "w") as logfile:
    rag_process = subprocess.Popen(["bash", "-c", f"source ~/.bashrc; set_capella_env && . ./start_rag_server.sh"], stdout=logfile, stderr=logfile)

llama_log = LOG_BASE_FILE + ".llama"
with open(llama_log, "w") as logfile:
    llama_process = subprocess.Popen(["bash", "-c", "source ~/.bashrc; set_scholarqabench_env && . ./start_llama_server_vllm.sh"], stdout=logfile, stderr=logfile)


while not check_server_health_request(port=RAG_PORT):
    print("Waiting for RAG server to be loaded!")
    time.sleep(2)

while not check_server_health_request(port=LLAMA_PORT):
    print("Waiting for Llama server to be loaded!")
    time.sleep(2)

name = RUN_NAME
if not USE_RAG:
    name += "_NO_RAG"
generate_output_dir = os.path.join(OUTPUT_BASE_DIR, name)
print(f"{name=}, {generate_output_dir=}")
generate_process = subprocess.Popen(["bash", "-c", f"source ~/.bashrc; set_capella_env && python generate_pubmedqa_output.py --topk 10 --output_dir {generate_output_dir}{' --use_rag' if USE_RAG else ''} {' --use_orig_ds --no-filter' if USE_ORIG_DS else ''} {' --test_set_only' if TEST_SET_ONLY else ''}"])
generate_process.wait()

stop_process(llama_process)
print("Waiting for Llama to stop")
llama_process.wait()

stop_process(rag_process)