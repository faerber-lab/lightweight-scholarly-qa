#!/bin/bash

uvicorn llama_server:app --host 127.0.0.1 --port 8000 --reload --reload-exclude=*.py --reload-include=llama_server.py