# Lightweight LLM-RAG Pipeline for Scientific Literature Comprehension

This repository provides a lightweight, modular Retrieval-Augmented Generation (RAG) pipeline designed to help understand and interact with scientific literature using Large Language Models (LLMs).

---

## 🔧 Finetuning

Finetuning scripts and configurations can be found in the [`os_train_data_finetune`](./os_train_data_finetune) directory.

---

## 🧩 LLM-RAG Pipeline Overview

This pipeline connects a local LLM with a retrieval component to enable question-answering over scientific texts. It is optimized for minimal resource usage and fast setup.

---

## Setup Instructions

### Required Libraries

Ensure the following libraries are installed in your virtual environment:

```bash
pip install langchain-community transformers sentence-transformers fastapi[standard] peft
```

> **Note:** The `[standard]` option for `fastapi` is required to include all necessary dependencies.

---

## Getting Started

The demonstration is run on unarXive datasets. All scripts for the pipeline can be found in the [`rag_unarxive`](./rag_unarxive) directory.

You'll need **two terminals** to start the backend servers and an optional **third terminal** to run the query interface.

### 1. Start the Llama Server

In the **first terminal**, activate your virtual environment and run:

```bash
./rag_unarxive/start_llama_server.sh
```

### 2. Start the RAG Server

In the **second terminal**, activate your virtual environment and run:

```bash
./rag_unarxive/start_rag_server.sh
```

Wait until both terminals show:

```
INFO:     Application startup complete.
```

### 3. Run the Demonstration

In a **third terminal**, activate the same virtual environment and run:

```bash
python rag_unarxive/RAG_openscholar_format.py
```

You’ll be prompted to enter a question. Type your query and interact with the system.

---

## Example Use Case

Ask domain-specific questions like:

```
What are the recent advances in transformer-based models for biomedical NLP?
```

And get responses grounded in your scientific corpus.

---

## 📁 Directory Structure (Optional)

```text
├── os_train_data_finetune/                 # Finetuning scripts and data handling
├── rag_unarxive/                           # RAG pipeline and server scripts
│   ├── start_llama_server.sh               # Script to launch the LLM server
│   ├── start_rag_server.sh                 # Script to launch the RAG server
│   ├── RAG_openscholar_format.py           # Entry point for querying the pipeline
│   ├── ....                               
├── README.md                               # This file

```

---

## 🧪 Notes

- Make sure your model and data paths are correctly configured in the scripts, for example in [`llama_pipeline.py`](./rag_unarxive/llama_pipeline.py):

```python
# Change this path to your local model location
model_path = "/data/horse/ws/s9650707-llm_workspace/scholaryllm_prot/os_train_data_finetune/model_checkpoints/..."
```

- The pipeline assumes local or pre-finetuned models and indexing.
