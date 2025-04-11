from enum import Enum
import json
import os
import pprint
import argparse
import re
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description="")
parser.add_argument("--topk", default=10, type=int)
parser.add_argument("--no-filter", action="store_true", default=False)
parser.add_argument("--use_rag", action="store_true", default=False)
parser.add_argument("--no_context", action="store_true", default=False)
parser.add_argument("--use_orig_ds", action="store_true", default=False)
parser.add_argument("--output_dir", default="./output_pubmedqa/", type=str)
args = parser.parse_args()

from datasets import load_dataset

from RAG_openscholar_format import chatbot
from classify_prompt import Task

from sklearn.metrics import f1_score


OUTPUT_DIR = args.output_dir


print(args)


OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.json")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"\n{OUTPUT_FILE=}\n")


if args.use_orig_ds:
    PUBMEDQA_CONFIG = 'pqa_labeled'

    #for config in ['pqa_artificial', 'pqa_labeled', 'pqa_unlabeled']:

    print(f"{PUBMEDQA_CONFIG=}")
    ds = load_dataset("qiaojin/PubMedQA", PUBMEDQA_CONFIG)["train"]

    if not args.no_filter:
        ds = ds.filter(lambda x: x['final_decision'] in ['yes', 'no'])
else:
    ds = []
    with open("/home/s9650707/s9650707-llm_secrets/datasets/scholarqabench/ScholarQABench/data/single_paper_tasks/pubmed_test.jsonl", 'r') as f:
        for line in f:
            ds.append(json.loads(line))

    print(ds[0])
    print(ds[-1])

print(f"{len(ds)=}")

total_count = len(ds)

class AnswerEnum(Enum):
    YES = 0
    NO = 1
    MAYBE = 2
    UNKNOWN = 3


def clean_answer(text: str) -> str:
    text = text.lower()
    text = re.sub('[^a-zA-Z]+', '', text)
    if text.startswith("yes"):
        return "yes"
    elif text.startswith("no"):
        return "no"
    elif text.startswith("maybe"):
        return "maybe"
    else:
        return text
#    if text == "yes":
#        return AnswerEnum.YES
#    elif text == "no":
#        return AnswerEnum.NO
#    elif text == "maybe":
#        return AnswerEnum.MAYBE
#    else:
#        return AnswerEnum.UNKNOWN

    

results = []
total_correct = 0
total_computed = 0
for i, data in tqdm(enumerate(ds), total=total_count):
    if args.use_orig_ds:
        question = data['question']
        contexts = {label: context for label, context in zip(data['context']['labels'], data['context']['contexts'])}
        long_answer = data['long_answer']
        final_decision = data['final_decision']
    else:
        question = data['input']
        # contexts = {label: context for label, context in zip(data['gold_ctx'][0]['text']['context']['labels'], data['context']['contexts'])}
        # contexts are in very inconsistent format... Let us follow the Openscholar paper and ignore the original
        # contexts...
        contexts = None
        #long_answer = data['long_answer']
        final_decision = data['answer']

    
    #pprint.pp({'id':id,'question':question, 'contexts':contexts, 'long_answer': long_answer, 'final_decision': final_decision})

    print("")
    if args.no_filter:
        task = Task.SINGLEQA_YESNOMAYBE
    else:
        task = Task.SINGLEQA_YESNO
    if args.no_context:
        golden_context=None
    else:
        golden_context=contexts
    result = chatbot(question, context=golden_context, fixed_task=task, continuous_chat=False, print_outputs=False, no_rag=not args.use_rag, no_clean_refs=True, rag_topk=args.topk, remove_gen_reflist=False, no_remove_after_excessive_linebreak=True, no_remove_cite_after_linebreak_or_dot=True)
    assert result is not None
    reply = result[0]
    references = result[1]
    orig_reply = result[2]
    #pprint.pp(result)
    if not args.use_rag:
        references.clear()
    
    answer = clean_answer(reply)
    correct = (answer == final_decision)
    total_correct += 1*correct
    total_computed += 1
    print(f"Answer is {'CORRECT' if correct else 'WRONG'}! ({answer=}, golden={final_decision})")
    if answer not in ['yes', 'no', 'maybe']:
        exit(0)

    result = {
        "input": question,
        "ctxs": [
            {
                "title": ref.title,
                "text": ref.cleaned_content(),

            }
            for ref in references
        ],
        "output": reply,
        "orig_output": orig_reply,
        "golden_answer": final_decision,
        "cleaned_answer": answer,
        "correct": correct,
        "use_rag": args.use_rag,
        "no_context": args.no_context,
        "no_filter": args.no_filter,
        "rag_topk": args.topk,
        "use_orig_ds": args.use_orig_ds
    }
    results.append(result)

    print(f"Running Total: so far {total_correct}/{total_computed} ({total_correct/total_computed:%}) are correct")

    print("")
print(results)

print(f"Total: {total_correct}/{total_computed} ({total_correct/total_computed:%}) are correct")


predictions = [_['cleaned_answer'] for _ in results]
groundtruth = [_['golden_answer'] for _ in results]

accuracy = np.mean([pred == gt for pred, gt in zip(predictions, groundtruth)])
f1 = f1_score(groundtruth, predictions, average='macro')

print(f"Results: \n")
print(f"Accuracy: {accuracy:%}")
print(f"F1 Score: {f1:%}")
results = {
    "data": results,
    "accuracy": accuracy,
    "f1": f1
}


with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f)