#!/usr/bin/env python
# coding: utf-8

print("Loading argparse, sys, re...")
import argparse
import pprint
import os
import sys
from enum import Enum
from references import References, clean_references, remove_after_excessive_linebreak, remove_cites_after_linebreak_or_dot, remove_generated_references

print("Loading llama_requests...")
from llama_request import llama_request

print("Loading others...")
from typing import Any, List, Tuple
print("Loading done...")


class Task(Enum):
    MultiQA = 0
    FollowUpQuestion = 1

def multi_qa_context(references: References, no_rag: bool=False) -> str:
    """Format retrieved documents into a context string for the MultiQA task."""
    
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
              "numbers should be added at the end of the sentences."
    
    if no_rag:
        context += "\n"
    else:
        context += "If no references  "\
              "are given, do not give an answer, instead point out that you could "\
              "not find any references!\nReferences:\n"
        context += "\nReferences:\n"
        context += references.format_for_context()

        #print(references.format_for_context())
        #pprint.pp(references)

    context += "Question: "

    return context


def generate_response(prompt: str, task: Task, initial: bool=False, previous_chat=None, references: References|None=None, no_rag: bool=False) -> Any:
    """Generate a response"""

    # If this is the first call, we potentially have to add system prompt or RAG context
    if initial:
        # Format context
        if task == Task.MultiQA:
            # check if references were found
            if no_rag or (references and len(references) > 0):
                context = multi_qa_context(references, no_rag)
                # Combine context and prompt
                full_prompt = context + prompt
                messages = [
                    {"role": "user", "content": full_prompt}
                ]
            else:
                return {'generated_text': [{'role': 'assistant', 'content': "Unfortunately, no references related to your question were found!"}]}
        else:
            system_prompt = "You are a helpful assistant. Answer the user's queries with highest attention to correctness. Be concise and give short answers, only adding strictly necessary detail. Do not write more than is asked."
            messages = [
                {
                    "role": "system", 
                    "content": system_prompt
                },
                {"role": "user", "content": prompt}
            ]
            
        #print(messages)
    else: # follow-up question
        full_prompt = prompt
        assert previous_chat is not None
        previous_chat.append({"role": "user", "content": full_prompt})
        messages = previous_chat
    
    chat = llama_request(messages, port=os.environ.get("LLAMA_PORT", "8004"))
    
    return chat



def chatbot(prompt: str|None, continuous_chat: bool=True, print_outputs=True, no_rag=False, no_clean_refs=False, rag_topk=10, remove_gen_reflist=True, no_remove_after_excessive_linebreak=False, no_remove_cite_after_linebreak_or_dot=False) -> None|Tuple[str, References, str]:
    #print("\nStarting ...")

    initial = True
    chat = None
    references = None

    
    # make interactive rag
    while True:
        if prompt is None or not initial:
            prompt = input("Ask your question. type 'quit' to exit. \nYou: ")
        if prompt == "quit":
            break
        if not prompt:
            continue
        
        # Until we have the task classification, we fix the task here
        task = Task.MultiQA

        # Generate and stream response
        references = None
        if initial:
            if task == Task.MultiQA:
                if no_rag:
                    references = References()
                else:
                    references = References.retrieve_from_vector_store(prompt, topk=rag_topk, port=int(os.environ.get("RAG_PORT", 8003)))
                    references.drop_refs_with_low_score(threshold=0.1)

            chat = generate_response(prompt, task=task, initial=initial, references=references, no_rag=no_rag)
        else:
            chat = generate_response(prompt, task=Task.FollowUpQuestion, initial=initial, references=references, previous_chat=chat['generated_text'], no_rag=no_rag)
        
        reply = chat['generated_text'][-1]['content']

        if references: 
            if not no_clean_refs:
                reply, references = clean_references(reply, references)
                references.update_from_semanticscholar()
            formatted_references = references.format_for_references()
        else:
            formatted_references = ""
        
        orig_reply = reply

        if remove_gen_reflist:
            reply = remove_generated_references(reply)
        
        if not no_remove_after_excessive_linebreak:
            reply = remove_after_excessive_linebreak(reply)
        
        if not no_remove_cite_after_linebreak_or_dot:
            reply = remove_cites_after_linebreak_or_dot(reply)

        #print(f"\n{remove_gen_reflist=}, {no_remove_after_excessive_linebreak=}, {no_remove_cite_after_linebreak_or_dot=}\n")

        chat['generated_text'][-1]['content'] = reply
        
        # Print the response (+ potentially References)
        if print_outputs:
            print("\nResponse:\n")
            print(reply)
            if len(formatted_references) > 0:
                print("\nReferences:\n")
                print(formatted_references)
            print("")
            print(orig_reply)
            print("")
        initial = False
        if not continuous_chat:
            return reply, references, orig_reply


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--no_clean_refs", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    chatbot(prompt=args.prompt, rag_topk=args.topk, no_clean_refs=args.no_clean_refs)



