from typing import Union

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from fastapi import FastAPI
from pydantic import BaseModel

from rag_utils import initialize_rag

import json
import os
import shutil

print("Initializing vector store...")
faiss_path = "/data/horse/ws/s9650707-llm_secrets/datasets/unarxive/faiss_index3"
#tmp_base_path = "/tmp/s9650707/faiss/"
#faiss_tmp_path = os.path.join(tmp_base_path, "unarxive/faiss_index3")

#if not os.path.exists(tmp_base_path):
#    os.makedirs(tmp_base_path)
#if not os.path.exists(faiss_tmp_path):
#    print("Copying faiss data to tmp")
#    shutil.copytree(faiss_path, faiss_tmp_path)

vector_store = initialize_rag(
    markdown_dir=None,
    faiss_index_file = faiss_path,
    load_index_from_file=True,
    store_index_to_file=False,
    max_files=None
)

app = FastAPI()


class RAGQuery(BaseModel):
    query: str
    k: int


@app.get("/")
def read_root():
    return {"Hello": "World 2"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.post("/rag_retrieve/")
def retrieve(rag_query: RAGQuery):
    try:
        return vector_store.similarity_search_with_relevance_scores(rag_query.query, k=rag_query.k)
    except Exception:
        return "FAIL"

@app.get("/healthy/")
def healthy():
    return {"status": "OK"}