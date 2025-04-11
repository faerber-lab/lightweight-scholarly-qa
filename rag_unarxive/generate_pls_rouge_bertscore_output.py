import json
import os
import pprint
import argparse
from tqdm import tqdm
from datasets import load_dataset
from pls_eval import compute_score_for_example

from RAG_openscholar_format import chatbot

promts_w_references_summarization = (
"Given an abstract of an academic paper and a set of passsages from relevant papers, generate a related work section summarizing relevant related work."
"Not all of the passages are relevant, so please carefully read the passages and only use passages that are related."  
"All of citation-worthy statements need to be supported by one of the references we provide as 'References' and appropriate citation numbers should be added at the last of the sentences."
"References should be formatted as [0], [1], [2], ..., [n]."
"Your answer should be marked as [Response_Start] and [Response_End]."
"Here's an example:\n##\n"
"References: \n{example_passages}"
"\nAbstract: {example_question}"
"\n[Response_Start]{example_answer}[Response_End]\nNow, please generate another related work given the following abstract.\n##\n")


prompts_w_references_summarization_zero_shot = ("Given an abstract of an academic paper and a set of passages from relevant papers, generate a related work section summarizing relevant related work. All of citation-worthy statements need to be supported by one of the references we provide as 'References' and appropriate citation numbers should be added at the last of the sentences. References should be formatted as [0], [1], [2], ..., [n].\nReferences: {context}\nAbstract: {input}")


print("Load scitldr dataset")
# Load the SciTLDR dataset from Hugging Face (AIC split, can also use Abstract, FullText not used in eval)
dataset = load_dataset("allenai/scitldr", "AIC", split="test")  # use "validation" or "test" if needed
#parts: ['train' (1.99k rows), 'validation' (619 rows), 'test' (618 rows)],
#features: ['source', 'source_labels', 'rouge_scores', 'paper_id', 'target'],

INPUT_FILE = "allenai/scitldr - AIC - test"
OUTPUT_DIR = "./output_pls_rouge_bertscore/"

parser = argparse.ArgumentParser(description="")
parser.add_argument("--output_dir", default=OUTPUT_DIR, type=str)
parser.add_argument("--topk", default=10, type=int)
parser.add_argument("--no_rag", action='store_true', default=False)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.json")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"\n{INPUT_FILE=}\n{OUTPUT_FILE}\n")
results = []
for i, inp in tqdm(enumerate(dataset), total=len(dataset)):
    print("")
    question = inp["input"]
    result = chatbot(
                question, 
                continuous_chat=False, 
                print_outputs=False, 
                no_rag=False,
                no_clean_refs=True, 
                rag_topk=args.topk, 
                remove_gen_reflist=False, 
                no_remove_after_excessive_linebreak=True, 
                no_remove_cite_after_linebreak_or_dot=True
    )
    assert result is not None
    reply = result[0]
    references = result[1]
    orig_reply = result[2]

    compute_score_for_example(source, targets, use_roberta_large=False):

    result = {
        "id": inp["id"],
        "input": inp["source"],
        "output": gen_sentence,
        "targets": targets,
        "llm_repeats": llm_repeats,
        "avg_scores": avg_scores,
        "max_scores": max_scores,
    }
    results.append(result)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    print("")
