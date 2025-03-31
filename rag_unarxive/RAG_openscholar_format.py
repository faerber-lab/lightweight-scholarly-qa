#!/usr/bin/env python
# coding: utf-8

print("Loading argparse, sys, re...")
import argparse
import sys
from enum import Enum
from references import References, clean_references

print("Loading llama_requests...")
from llama_request import llama_request

print("Loading others...")
from typing import Any, List, Tuple
print("Loading done...")


class Task(Enum):
    MultiQA = 0
    FollowUpQuestion = 1

def multi_qa_context(references: References) -> str:
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
              "numbers should be added at the end of the sentences. If no references  "\
              "are given, do not give an answer, instead point out that you could "\
              "not find any references!\nReferences:\n"
    
    context += references.format_for_context()

    context += "Question: "

    return context


def generate_response(prompt: str, task: Task, initial: bool=False, previous_chat=None, references: References|None=None) -> Any:
    """Generate a response"""

    # If this is the first call, we potentially have to add system prompt or RAG context
    if initial:
        # Format context
        if task == Task.MultiQA:
            # check if references were found
            if references and len(references) > 0:
                context = multi_qa_context(references)
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
    
    chat = llama_request(messages, port="8004")
    
    return chat



def chatbot(prompt: str|None, continuous_chat: bool=True, print_outputs=True) -> None|Tuple[str, References]:
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
                references = References.retrieve_from_vector_store(prompt, topk=10, port=8003)
                references.drop_refs_with_low_score(threshold=0.1)

            chat = generate_response(prompt, task=task, initial=initial, references=references)
        else:
            chat = generate_response(prompt, task=Task.FollowUpQuestion, initial=initial, references=references, previous_chat=chat['generated_text'])
        
        reply = chat['generated_text'][-1]['content']

        if references: 
            reply, references = clean_references(reply, references)
            references.update_from_semanticscholar()
            formatted_references = references.format_for_references()
        else:
            formatted_references = ""

        chat['generated_text'][-1]['content'] = reply
        
        # Print the response (+ potentially References)
        if print_outputs:
            print("\nResponse:\n")
            print(reply)
            if len(formatted_references) > 0:
                print("\nReferences:\n")
                print(formatted_references)
            print("")
        initial = False
        if not continuous_chat:
            return reply, references


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()
    chatbot(args.prompt)



