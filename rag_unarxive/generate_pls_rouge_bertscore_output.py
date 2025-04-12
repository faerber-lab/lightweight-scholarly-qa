import json
import os
import re
import pprint
import argparse
from tqdm import tqdm
from classify_prompt import Task
from datasets import load_dataset
from pls_eval import compute_score_for_example
from RAG_openscholar_format import generate_response

print("Load scitldr dataset")
# Load the SciTLDR dataset from Hugging Face (AIC split, can also use Abstract, FullText not used in eval)
dataset = load_dataset("allenai/scitldr", "AIC", split="test")  # use "validation" or "test" if needed
#parts: ['train' (1.99k rows), 'validation' (619 rows), 'test' (618 rows)],
#features: ['source', 'source_labels', 'rouge_scores', 'paper_id', 'target'],

INPUT_FILE = "allenai/scitldr - AIC - test"
OUTPUT_DIR = "./output_pls_rouge_bertscore/"

parser = argparse.ArgumentParser(description="")
parser.add_argument("--output_dir", default=OUTPUT_DIR, type=str)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.json")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"\n{INPUT_FILE=}\n{OUTPUT_FILE}\n")
results = []
#dataset = dataset.select(range(10)) # use only part for debug
for ex in tqdm(dataset, desc="Computing SCORES"):
    print("")
    # NOTE: IMPORTANT: source should be the generated sum sentence for eval
    #source = ex["source"]
    chat = generate_response(" ".join(ex['source']), task=Task.SUMMARIZATION_SCITLDR, initial=True)
    assert chat is not None
    source = chat['generated_text'][1]['content']
    try:
        intro_str = re.search('(Here.*:\\n\\n)', source).group(1)
        source = source.replace(intro_str, "")
    except:
        pass

    targets = ex["target"]
    eid = ex.get("paper_id", "N/A")

    avg_scores, max_scores = compute_score_for_example(source, targets)
    results.append({
        "id": eid,
        "avg_scores": avg_scores,
        "max_scores": max_scores,
        "input": ex["source"],
        "output": source,
        "targets": targets,
        "chat": chat,
    })

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    print("")

# average over all papers
avg_max_scores = {}
avg_avg_scores = {}
for key in results[0]['max_scores'].keys():
    avg_max_scores[key] = {}
    avg_avg_scores[key] = {}
    for metric in ['precision', 'recall', 'fmeasure']:
        avg_max_scores[key][metric] = sum(r['max_scores'][key][metric] for r in results) / len(results)
        avg_avg_scores[key][metric] = sum(r['avg_scores'][key][metric] for r in results) / len(results)

results.append({
    "id": "all_examples",
    "avg_scores": avg_avg_scores,
    "max_scores": avg_max_scores,
    "input": None,
    "output": None,
    "targets": None,
    "chat": None,
})

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f)
print("")

