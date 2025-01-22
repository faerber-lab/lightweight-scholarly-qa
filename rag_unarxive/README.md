# RAG demonstration on unarXive

## Setup
### Libraries
Install all required libraries into your virtual environment:
- `langchain-community`
- `transformers`
- `fastapi[standard]`    # the `[standard]` is necessary
- `peft`

### Startup
You need to start both the Llama server and the RAG server, for this use two terminals and load your virtual environment in both.
- Run `start_llama_server.sh` in the first terminal
- Run `start_rag_server.sh` in the second terminal

### Usage
Open a third terminal, load your virtual environment. Wait until both other terminals show `INFO:     Application startup complete.`
- Run `python RAG.py`
- Enter your question
- enjoy