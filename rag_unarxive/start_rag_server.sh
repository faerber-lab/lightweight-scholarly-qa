#!/bin/bash

RAG_PORT=${RAG_PORT:-8003}
echo "RAG_PORT=$RAG_PORT"
uvicorn rag_server:app --host 127.0.0.1 --port $RAG_PORT --reload --reload-exclude=*.py --reload-include=rag_server.py
