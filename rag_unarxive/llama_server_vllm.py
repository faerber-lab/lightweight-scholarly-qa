from typing import Union

from fastapi import FastAPI, Request
from pydantic import BaseModel


import json
import os
import shutil

import argparse

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

#from llama_pipeline import get_llama_pipeline


app = FastAPI()

#base_model = "meta-llama/Llama-3.1-8B-Instruct"
#base_model = "meta-llama/Llama-3.2-3B-Instruct"
base_model = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

# base model
#lora_model=None
# 8k context Finetuned model
#lora_model = "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_lora32_shuffled_split/checkpoint-7970/"
# 8k context 2 epoch Finetuned model
#lora_model = "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_lora32_shuffled_split_8k_context_2_epoch/checkpoint-3984/"
# 16k context Finetuned model
#lora_model = "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_lora32_shuffled_split_16k_context/checkpoint-3985/"
# 16k context 2 epoch Finetuned model
#lora_model = "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints_100percent_lora32_shuffled_split_16k_context_2_epoch_dist_train/checkpoint-1992/"

lora_model = os.environ.get("LORA_MODEL", None)

print(f"{base_model=}\n{lora_model=}")
llm = LLM(model=base_model, enable_lora=True, max_model_len=16384, max_lora_rank=32)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=3000, frequency_penalty=1.2, n=1, best_of=6)

if lora_model is None or lora_model == "":
    openscholar_lora_request = None
else:
    openscholar_lora_request = LoRARequest("openscholar_lora", 1, lora_model, long_lora_max_len=16384)

#class RAGQuery(BaseModel):
#    query: str
#    k: int


@app.post("/llama_generate/")
async def llama_generate(request: Request):
    data = await request.json()

    outputs = llm.chat(data, sampling_params, lora_request=openscholar_lora_request)
    data.append({"role": "assistant", "content": outputs[0].outputs[0].text})
    return {"generated_text": data}

    return outputs
    #return pipeline(data, do_sample=True, max_new_tokens=1024, temperature=0.7, top_p=0.9, repetition_penalty=1.2, length_penalty=2.0, num_beams=6)[0]


@app.get("/healthy/")
def healthy():
    return {"status": "OK"}