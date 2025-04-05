import json
import os
import pprint
import argparse
from tqdm import tqdm

from RAG_openscholar_format import chatbot

SCHOLARQABENCH_DIR = "/data/horse/ws/s9650707-llm_secrets/datasets/scholarqabench/ScholarQABench/data/scholarqa_multi/"
INPUT_FILE = os.path.join(SCHOLARQABENCH_DIR, "human_answers.json")

OUTPUT_DIR = "./output_scholarqabench_multiqa/"

parser = argparse.ArgumentParser(description="")
parser.add_argument("--output_dir", default=OUTPUT_DIR, type=str)
parser.add_argument("--no_rag", action='store_true', default=False)
args = parser.parse_args()

OUTPUT_DIR = args.output_dir


OUTPUT_FILE = os.path.join(OUTPUT_DIR, "output.json")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


print(f"\n{INPUT_FILE=}\n{OUTPUT_FILE}\n")

with open(INPUT_FILE, "r") as f:
    input_data = json.load(f)

total_count = len(input_data)



results = []
for i, inp in tqdm(enumerate(input_data), total=total_count):
    print("")
    question = inp["input"]
    result = chatbot(question, continuous_chat=False, print_outputs=False, no_rag=args.no_rag, no_clean_refs=True)
    assert result is not None
    reply = result[0]
    references = result[1]
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
        "output": reply
    }
    results.append(result)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f)
    print("")
