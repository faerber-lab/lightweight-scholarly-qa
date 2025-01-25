#!/usr/bin/env python
# coding: utf-8



print("Loading rag_requests...")
from rag_request import rag_request
print("Loading llama_requests...")
from llama_request import llama_request

print("Loading others...")
from typing import List
from langchain.docstore.document import Document
print("Loading done...")


SYSTEM_PROMPT_INITIAL_GENERATION = """
You are a very competent and helpful scholarly AI assistant and an expert in most scholarly disciplines.
Your task is to answer questions about scientific topics based on reference information from scientific papers that have been uploaded to arXiv.
You will be given several potentially relevant sections several papers, listed with their file name (doc_00000000.md etc., which you should ignore), and their title.
If the references contain no useful information, answer 'The references contain no information on your question!'
Here are the reference sections:
"""



def retrieve_formatted_documents(prompt: str, vector_store) -> str:        
    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(prompt, k=10)
    
    # Format context
    context = format_context(retrieved_docs)
    return context

def format_context(retrieved_docs: List[Document]) -> str:
    """Format retrieved documents into a context string."""
    context = "Reference information:\n"
    for doc, score in retrieved_docs:
        content = doc["page_content"]
        source = doc["metadata"].get("source", "Unknown")
        header = doc["metadata"].get("header", "")
        
        context += f"\n--- From {source}"
        if header:
            context += f" ({header})"
        context += f" ---\n{content}\n"
    
    context += "\nBased on the above information, please answer: "
    return context

def generate_response(prompt: str, initial=False, previous_chat=None):
    """Generate a streaming response using RAG and the fine-tuned model."""
    if not prompt:
        return "Hi I am an assistant for Candulor GmbH. I can help you with questions about their products. What do you need help with?"
    
    # Retrieve relevant documents - changed from k=3 to k=5
    if initial:
        retrieved_docs = rag_request(prompt, k=10, port=8001)
    
        # Format context
        context = format_context(retrieved_docs)
        
        # Combine context and prompt
        full_prompt = context + prompt
    else:
        full_prompt = prompt
        
    if previous_chat is None:
        messages = [
            {
                "role": "system", 
                "content": SYSTEM_PROMPT_INITIAL_GENERATION
            },
            {"role": "user", "content": full_prompt}
        ]
    else:
        print(type(previous_chat))
        previous_chat.append({"role": "user", "content": full_prompt})
        messages = previous_chat
    
    chat = llama_request(messages, port=8000)
    
    return chat



def rag():
    print("\nStarting ...")

    initial = True
    chat = None

    
    # make interactive rag
    while True:
        prompt = input("Ask your question. type 'quit' to exit. \nYou: ")
        if prompt == "quit":
            break
        if not prompt:
            print("Usage: python RAG.py <prompt>")
            sys.exit(1)
        # Generate and stream response
        if initial:
            chat = generate_response(prompt, initial=initial)
        else:
            chat = generate_response(prompt, initial=initial, previous_chat=chat['generated_text'])
        reply = chat['generated_text'][-1]['content']
        print("\nResponse:\n")
        print(reply)
        print("")
        initial = False


rag()



