#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 s1304873 <s1304873@c125>
#
# Distributed under terms of the MIT license.

"""

"""

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from datasets import load_dataset
from rouge_score import rouge_scorer
from evaluate import load
from tqdm import tqdm


# initialize BertScorer and ROUGE scorer
bertscore = load("bertscore")
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_score_for_example(source, targets, use_roberta_large=False):
    if source is not str:
        source = " ".join(source)

    # eval BertScore: compare against all gold sentences and get maximum
    scores = []
    for gold_sentence in targets:
        # eval rouge and turn into dict
        rouge_score = rouge_scorer.score(source, gold_sentence)
        for key in rouge_score.keys():
            rouge_score[key] = rouge_score[key]._asdict()

        # eval bertscore
        if use_roberta_large:
            # use model 'roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.51.2)'
            bert_score = bertscore.compute(predictions=[source], references=[target], lang="en")
        else:
            # use model 'distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.51.2)'
            bert_score = bertscore.compute(predictions=[source], references=[gold_sentence], model_type="distilbert-base-uncased")

        bert_score = {'bertscore': {
                'precision' : bert_score['precision'][0],
                'recall'    : bert_score['recall'][0],
                'fmeasure'  : bert_score['f1'][0],
            }
        }

        # append to results
        scores.append(rouge_score | bert_score)

    # Max/Average across all target summaries
    max_scores = {}
    avg_scores = {}

    for key in scores[0].keys():
        max_scores[key] = {}
        avg_scores[key] = {}
        for metric in ['precision', 'recall', 'fmeasure']:
            max_scores[key][metric] = max(s[key][metric] for s in scores)
            avg_scores[key][metric] = sum(s[key][metric] for s in scores) / len(scores)

    return avg_scores, max_scores


if __name__ == "__main__":
    print("Load scitldr dataset")
    # Load the SciTLDR dataset from Hugging Face (AIC split, can also use Abstract, FullText not used in eval)
    dataset = load_dataset("allenai/scitldr", "AIC", split="test")  # use "validation" or "test" if needed
    #parts: ['train' (1.99k rows), 'validation' (619 rows), 'test' (618 rows)],
    #features: ['source', 'source_labels', 'rouge_scores', 'paper_id', 'target'],

    # eval AIC example papers
    dataset = dataset.select(range(10)) # use only part 
    results = []
    print("Evaluate ROUGE and BertScore for each example and get max/average from all gold summarization sentences (author, peer-reviewer)")
    for ex in tqdm(dataset, desc="Computing SCORES"):
        # NOTE: IMPORTANT: source should be the generated sum sentence for eval
        # source = chatbot
        source = ex["source"]
        targets = ex["target"]
        eid = ex.get("paper_id", "N/A")
    
        avg_scores, max_scores = compute_score_for_example(source, targets)
        results.append({
            "eid": eid,
            "avg_scores": avg_scores,
            "max_scores": max_scores,
        })
    
    # Print a few example results
    print("-" * 40)
    print("Average")
    print("-" * 40)
    for r in results[:5]:
        print(f"Example ID: {r['eid']}")
        for metric, scores in r['avg_scores'].items():
            print(f"  {metric.upper()}: P={scores['precision']:.4f}, R={scores['recall']:.4f}, F1={scores['fmeasure']:.4f}")
        print("-" * 40)
    
    print("-" * 40)
    print("Maximum")
    print("-" * 40)
    for r in results[:5]:
        print(f"Example ID: {r['eid']}")
        for metric, scores in r['max_scores'].items():
            print(f"  {metric.upper()}: P={scores['precision']:.4f}, R={scores['recall']:.4f}, F1={scores['fmeasure']:.4f}")
        print("-" * 40)
    
