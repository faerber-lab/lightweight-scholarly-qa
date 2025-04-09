#!/bin/bash

LLAMA_PORT=${LLAMA_PORT:-8004}
echo "LLAMA_PORT=$LLAMA_PORT"
uvicorn llama_server_vllm:app --host 127.0.0.1 --port $LLAMA_PORT --reload --reload-exclude=*.py --reload-include=llama_*.py
