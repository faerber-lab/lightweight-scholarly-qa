#!/bin/bash

uvicorn rag_server:app --host 127.0.0.1 --port 8001 --reload --reload-exclude=*.py --reload-include=rag_server.py