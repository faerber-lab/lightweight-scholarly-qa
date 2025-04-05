from typing import Union

from fastapi import FastAPI, Request
from pydantic import BaseModel


import json
import os
import shutil

from llama_pipeline import get_llama_pipeline


app = FastAPI()

pipeline = get_llama_pipeline()

#class RAGQuery(BaseModel):
#    query: str
#    k: int


@app.post("/llama_generate/")
async def llama_generate(request: Request):
    data = await request.json()
    return pipeline(data, do_sample=True, max_new_tokens=1024, temperature=0.7, top_p=0.9, repetition_penalty=1.2, length_penalty=2.0, num_beams=6)[0]


@app.get("/healthy/")
def healthy():
    return {"status": "OK"}