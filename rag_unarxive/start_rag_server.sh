#!/bin/bash

uvicorn rag_server:app --host 127.0.0.1 --port 8003 --reload --reload-exclude=*.py --reload-include=rag_server.py
