from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
import torch
from threading import Thread
import sys
from rag_utils import initialize_rag
from typing import List
from langchain.docstore.document import Document



print("Initializing vector store...")
vector_store = initialize_rag(
    markdown_dir="/data/horse/ws/s9650707-llm_secrets/datasets/unarxive/md3/",
    faiss_index_file = "/data/horse/ws/s9650707-llm_secrets/datasets/unarxive/faiss_index3",
    load_index_from_file=False,
    store_index_to_file=True,
    max_files=None,
    docs_per_batch=262144
)
