#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 s1304873 <s1304873@c2>
#
# Distributed under terms of the MIT license.

"""
For train rouge_scores are almost as big as source (sentences), for test this is not the case...
"""

from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm

# Load the SciTLDR dataset from Hugging Face (AIC split)
dataset = load_dataset("allenai/scitldr", "AIC", split="test")  # use "validation" or "test" if needed

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Function to compute average ROUGE for one example
def compute_rouge_for_example(source_sents, targets):
    source_text = " ".join(source_sents)
    scores = []

    for target in targets:
        score = scorer.score(source_text, target)
        scores.append(score)

    # Average across all target summaries
    avg_scores = {}
    max_scores = {}
    for key in scores[0].keys():
        avg_scores[key] = {
            metric: sum(s[key][metric] for s in scores) / len(scores)
            for metric in [0, 1, 2]
            #for metric in ['precision', 'recall', 'fmeasure']
        }
        max_scores[key] = {
            metric: max(s[key][metric] for s in scores)
            for metric in [0, 1, 2]
            #for metric in ['precision', 'recall', 'fmeasure']
        }
    return avg_scores, max_scores

# Compute ROUGE scores for the dataset
results = []

for ex in tqdm(dataset, desc="Computing ROUGE"):
    source = ex["source"]
    targets = ex["target"]
    eid = ex.get("paper_id", "N/A")

    avg_rouge_scores, max_rouge_scores = compute_rouge_for_example(source, targets)
    results.append({
        "eid": eid,
        "avg_rouge": avg_rouge_scores,
        "max_rouge": max_rouge_scores,
    })

# Print a few example results
print("-" * 40)
print("Average")
print("-" * 40)
for r in results[:5]:
    print(f"Example ID: {r['eid']}")
    for metric, scores in r['avg_rouge'].items():
        print(f"  {metric.upper()}: P={scores[0]:.4f}, R={scores[1]:.4f}, F1={scores[2]:.4f}")
        #print(f"  {metric.upper()}: P={scores['precision']:.4f}, R={scores['recall']:.4f}, F1={scores['fmeasure']:.4f}")
    print("-" * 40)

print("-" * 40)
print("Maximum")
print("-" * 40)
for r in results[:5]:
    print(f"Example ID: {r['eid']}")
    for metric, scores in r['max_rouge'].items():
        print(f"  {metric.upper()}: P={scores[0]:.4f}, R={scores[1]:.4f}, F1={scores[2]:.4f}")
        #print(f"  {metric.upper()}: P={scores['precision']:.4f}, R={scores['recall']:.4f}, F1={scores['fmeasure']:.4f}")
    print("-" * 40)

