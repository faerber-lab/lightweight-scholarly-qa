#!/bin/bash

uvicorn llama_server:app --host 127.0.0.1 --port 8002 --reload --reload-exclude=*.py --reload-include=llama_*.py
