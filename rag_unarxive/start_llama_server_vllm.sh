#!/bin/bash

uvicorn llama_server_vllm:app --host 127.0.0.1 --port 8004 --reload --reload-exclude=*.py --reload-include=llama_*.py
