import json
import os
import pprint
import argparse
from tqdm import tqdm

from datasets import load_dataset

from RAG_openscholar_format import chatbot


OUTPUT_DIR_WITH_RAG = "./output_pubmedqa_rag/"
OUTPUT_DIR_WITHOUT_RAG = "./output_pubmedqa_no_rag/"

parser = argparse.ArgumentParser(description="")
parser.add_argument("--topk", default=10, type=int)
parser.add_argument("--no-filter", action="store_true", default=False)
parser.add_argument("--use_rag", action="store_true", default=False)
args = parser.parse_args()

print(args)


OUTPUT_FILE_WITH_RAG = os.path.join(OUTPUT_DIR_WITH_RAG, "output.json")
OUTPUT_FILE_WITHOUT_RAG = os.path.join(OUTPUT_DIR_WITHOUT_RAG, "output.json")

if not os.path.exists(OUTPUT_DIR_WITH_RAG):
    os.makedirs(OUTPUT_DIR_WITH_RAG)
if not os.path.exists(OUTPUT_DIR_WITHOUT_RAG):
    os.makedirs(OUTPUT_DIR_WITHOUT_RAG)


print(f"\n{OUTPUT_FILE_WITH_RAG=}\n{OUTPUT_FILE_WITHOUT_RAG=}\n")

PUBMEDQA_CONFIG = 'pqa_labeled'

#for config in ['pqa_artificial', 'pqa_labeled', 'pqa_unlabeled']:

print(f"{PUBMEDQA_CONFIG=}")
ds = load_dataset("qiaojin/PubMedQA", PUBMEDQA_CONFIG)["train"]

if not args.no_filter:
    ds = ds.filter(lambda x: x['final_decision'] in ['yes', 'no'])


print(f"{len(ds)=}")

total_count = len(ds)

results = []
for i, data in tqdm(enumerate(ds), total=total_count):
    if i>10:
        break
    id = data['pubid']
    question = data['question']
    contexts = {label: context for label, context in zip(data['context']['labels'], data['context']['contexts'])}
    long_answer = data['long_answer']
    final_decision = data['final_decision']
    
    pprint.pp({'id':id,'question':question, 'contexts':contexts, 'long_answer': long_answer, 'final_decision': final_decision})

    print("")
    result = chatbot(question, continuous_chat=False, print_outputs=False, no_rag=args.no_rag, no_clean_refs=True, rag_topk=args.topk, remove_gen_reflist=False, no_remove_after_excessive_linebreak=True, no_remove_cite_after_linebreak_or_dot=True)
    assert result is not None
    reply = result[0]
    references = result[1]
    orig_reply = result[2]
    if args.no_rag:
        references.clear()

    result = {
        "input": question,
        "ctxs": [
            {
                "title": ref.title,
                "text": ref.cleaned_content(),

            }
            for ref in references
        ],
        "annotator": inp["annotator"],
        "id": inp["id"],
        "subject": inp["subject"],
        "output": reply,
        "orig_output": orig_reply
    }
    results.append(result)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    print("")
