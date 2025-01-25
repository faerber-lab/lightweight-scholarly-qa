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
Please use citations to the references used in your answer in the style [1],.. After your answer, provide a 
table of references in the format [1] Author a, Author b,... "This is the title of the paper", published in ... 
Here are the reference sections:
"""

SYSTEM_PROMPT_REVIEWER = """
You are a very competent and helpful, but strict reviewer for text that a scholarly AI assistant wrote. You are also an expert in most scholarly disciplines.
The assistant is supposed to answer questions based on reference sections from scientific papers on arXiv. It is given several potentially relevant sections
from several papers and should write a correct answer with citations from that.
Your job is give feedback on how to improve the answers, mainly focusing on these points:
- factuality: are the facts supported by the reference sections? Is a table of references given? Are the references correct and support the answers?
- references: Are citations used in the answer in correct positions? Are all citations from the table of references used? Do all citations in the answer have an entry in the table? Are references duplicated?
- language: is it understandable language?
- conciseness: contains all important details, but is free from unnecessary detail?
- relevance: is the answer relevant to the question?
- completeness: Is anything missing from the answer? Is the question really completely answered?
It is of utmost importance that you give clear and concise feedback with proposals of how to improve the text.
Here are the reference sections:
"""

#, with to rate the answers on a scale from 1-5 where 1 is the worst and 5 is the best on the following points:


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

def generate_review(full_prompt: str, answer: str):
    full_prompt = full_prompt + "\nThe original answer was:\n" + answer + "\nPlease write your review now!\n"
    messages = [
        {
            "role": "system", 
            "content": SYSTEM_PROMPT_REVIEWER
        },
        {"role": "user", "content": full_prompt}
    ]
    
    chat = llama_request(messages, port=8000)
    return chat

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

def improve_responese(previous_chat, review):
    messages = previous_chat
    messages.append(review[-1])
    messages[-1]['role'] = 'user'
    messages[-1]['content'] += "\nPlease incorporate this review and improve your anser! Do not tell what you improved, only give the revised answer!\n"

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
            orig_answer = chat['generated_text'][-1]['content']
            review = generate_review(chat['generated_text'][1]['content'], chat['generated_text'][2]['content'])
            review_answer = review['generated_text'][-1]['content']
            improved_chat = improve_responese(chat['generated_text'], review['generated_text'])
            improved_answer = improved_chat['generated_text'][-1]['content']
            print("\n# Response:\n")
            print(orig_answer)
            print("\n")
            print("\n# Review:\n")
            print(review_answer)
            print("\n")
            print("\n# Improved answer:\n")
            print(improved_answer)
            review = generate_review(chat['generated_text'][1]['content'], chat['generated_text'][-1]['content'])
            review_answer = review['generated_text'][-1]['content']
            improved_chat = improve_responese(chat['generated_text'], review['generated_text'])
            improved_answer = improved_chat['generated_text'][-1]['content']
            print("\n# Review:\n")
            print(review_answer)
            print("\n")
            print("\n# Improved answer:\n")
            print(improved_answer)
            initial = False
        else:
            improved_chat = generate_response(prompt, initial=initial, previous_chat=improved_chat['generated_text'])
            reply = improved_chat['generated_text'][-1]['content']
            print(reply)
        

rag()



