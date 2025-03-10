#!/usr/bin/env python
# coding: utf-8

print("Loading argparse, sys, re...")
import argparse
import sys
import re
import semanticscholar

print("Loading rag_requests...")
from rag_request import rag_request
print("Loading llama_requests...")
from llama_request import llama_request

print("Loading others...")
from typing import List, Tuple
from langchain.docstore.document import Document
print("Loading done...")



def retrieve_formatted_documents(prompt: str, vector_store) -> str:        
    # Retrieve relevant documents
    retrieved_docs = vector_store.similarity_search(prompt, k=6)
    
    # Format context
    context = format_context(retrieved_docs)
    return context

def format_context(retrieved_docs: List[Document]) -> Tuple[str, str]:
    """Format retrieved documents into a context string."""
    
    # [sic!] 
    context = "Answer the following question related to the recent research. " \
              "Your answer should be detailed and informative, and is likely " \
              "to be more than one paragraph. Your answer should be horistic, "  \
              "based on multiple evidences and references, rather than a short " \
              "sentence only based on a single reference. Make the answer well-" \
              "structured, informative so that real-world scientists can get a " \
              "comprehensive overview of the area based on your answer. All of " \
              "citation-worthy statements need to be supported by one of the " \
              "references we provide as 'References' and appropriate citation " \
              "numbers should be added at the end of the sentences. If no references " \
              "are given, do not give an answer, instead point out that you could "\
              "not find any references!\nReferences:\n"

    references = []

    used_idx=0
    
    for i, (doc, score) in enumerate(retrieved_docs):
        content = doc["page_content"]
        content_split = content.split("\n")
        main_content = " ".join(content_split[4:])
        #source = doc["metadata"].get("source", "Unknown")
        #header = doc["metadata"].get("header", "")
        
        #context += f"\n--- From {source}"
        #if header:
        #    context += f" ({header})"
        #context += f" ---\n{content}\n"
        authors = doc["metadata"].get("authors", "unknown authors")
        title = doc["metadata"].get("title", "unknown title")
        published_in = doc["metadata"].get("published_in", "")
        doi = doc["metadata"].get("doi", "")
        print(title, score)
        if score < 0.25:
            continue
        context += f"[{used_idx+1}] {main_content}\n"
        references.append({"title": title, "authors": authors, "venue": published_in, "url": doi})
        used_idx += 1
    if used_idx == 0:
        context += "No references found!\n"
    context = context.replace(" {{cite:}}", "")
    context += "Question: "
    return context, references

def generate_response(prompt: str, initial=False, previous_chat=None):
    """Generate a streaming response using RAG and the fine-tuned model."""
    if not prompt:
        return "Hi I am an assistant for Candulor GmbH. I can help you with questions about their products. What do you need help with?"
    references = None
    # Retrieve relevant documents - changed from k=3 to k=5
    if initial:
        retrieved_docs = rag_request(prompt, k=10, port=8003)
    
        # Format context
        context, references = format_context(retrieved_docs)
        
        # Combine context and prompt
        full_prompt = context + prompt
    else:
        full_prompt = prompt

    print(full_prompt)
        
    if previous_chat is None:
        messages = [
            #{
            #    "role": "system", 
            #    "content": SYSTEM_PROMPT_INITIAL_GENERATION
            #},
            {"role": "user", "content": full_prompt}
        ]
    else:
        print(type(previous_chat))
        previous_chat.append({"role": "user", "content": full_prompt})
        messages = previous_chat
    
    chat = llama_request(messages, port=8002)
    
    print(chat)
    #if references is not None:
    #    chat['generated_text'][-1]["content"] += (f"\n\n==================\n\nReferences:\n\n{references}")

    if len(references) == 0:
        chat['generated_text'][-1]["content"] = "Unfortunately, no references related to your question were found!"
    
    return chat, references, retrieved_docs if references else None


def remove_unused_references(reply, references):
    return reply, references

def uniquify_and_renumber_references(reply, references):
    """
    references contain "authors", "title", "venue" and "url"
    """
    replace = []
    previous_refs = {}
    running_idx = 1

    titles = [ref["title"] for ref in references]

    # find duplicate references, compute the new indices for the 
    # remaining unified references.
    for i, (title, ref) in enumerate(zip(titles, references)):
        #print(i,title)
        prev_idx = i+1
        if title in previous_refs:
            new_idx = previous_refs[title]["idx"]
            replace.append((prev_idx, new_idx))
        else:
            previous_refs[title] = {"idx": running_idx, "ref": ref}
            if running_idx != prev_idx:
                replace.append((prev_idx, running_idx))
            running_idx += 1
            
    #print("\n==========\nbefore replacing:\n", reply)
    # replace the old reference indices in the reply by the new ones
    for prev, repl in replace:
        reply = reply.replace(f"[{prev}]", f"[{repl}]")
    #print("\n==========\nafter replacing:\n", reply, "\n==========\n")
    #print(replace)
    #print(previous_refs)

    new_refs = []
    for title, val in previous_refs.items():
        new_refs.append(val['ref'])
                
    return reply, new_refs

def format_references(references):
    for i in range(len(references)):
        title = references[i]['title']
        semanticscholar_result = semanticscholar.fetch_paper_from_semantic_scholar(title)
        res = semanticscholar_result
        # if we cannot find the reference on semanticscholar, we still use our own info which is usually worse,
        # but it is what we have...
        if res is None:
            res = references[i]
        authors = res["authors"]
        title = res["title"]
        venue = res["venue"]
        url = res["url"]
        references[i] = f"{authors}. \"{title}\". {venue}, {url}"
        references[i].replace(", ,", "")
        references[i].replace("..", "")
    ref_str = "\n".join(f"[{i+1}] {ref}" for i, ref in enumerate(references))
    return ref_str
    
def clean_citations(text):
    def process_match(match):
        citations = re.findall(r'\d+', match.group())  # Extract numbers from the matched citations
        unique_sorted_citations = sorted(set(map(int, citations)))  # Remove duplicates and sort
        return ''.join(f'[{num}]' for num in unique_sorted_citations)  # Reconstruct sorted citations

    # Replace all citation groups with cleaned versions
    cleaned_text = re.sub(r'(\[(\d+)\])+', process_match, text)
    return cleaned_text

def remove_generated_references(text):
    split_text = text.split("References:")
    return split_text[0]

def rag(prompt: str|None):
    print("\nStarting ...")

    initial = True
    chat = None

    
    # make interactive rag
    while True:
        if prompt is None or not initial:
            prompt = input("Ask your question. type 'quit' to exit. \nYou: ")
        if prompt == "quit":
            break
        if not prompt:
            print("Usage: python RAG.py <prompt>")
            sys.exit(1)
        # Generate and stream response
        if initial:
            chat, references, retrieved_docs = generate_response(prompt, initial=initial)
        else:
            chat, references, retrieved_docs = generate_response(prompt, initial=initial, previous_chat=chat['generated_text'])
        reply = chat['generated_text'][-1]['content']

        reply, references = remove_unused_references(reply, references)
        reply, references = uniquify_and_renumber_references(reply, references)
        formatted_references = format_references(references)

        #print("\n=======\nBefore cleaning\n=========\n")
        #print(reply)
        reply = clean_citations(reply)
        reply = remove_generated_references(reply)
        #print("\n=======\After cleaning\n=========\n")
        #print(reply)
        #print("\n=========")

        reply = chat['generated_text'][-1]['content'] = reply
        
        print("\nResponse:\n")
        print(reply)
        if len(formatted_references) > 0:
            print("\nReferences:\n")
            print(formatted_references)
        print("")
        initial = False


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()
    rag(args.prompt)



