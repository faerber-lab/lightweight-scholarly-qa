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


# load scitldr
print("Load scitldr dataset")
#ds = load_dataset("allenai/scitldr", "FullText")
ds = load_dataset("allenai/scitldr", "AIC")
#parts: ['train' (1.99k rows), 'validation' (619 rows), 'test' (618 rows)],
#features: ['source', 'source_labels', 'rouge_scores', 'paper_id', 'target'],

source_l = ds['test'][0]['source']
target_l = ds['test'][0]['target']

# concat
source0 = source_l[0]
target0 = target_l[0]
source = ' '.join(source_l)
target = ' '.join(target_l)

# eval BertScore
print("Evaluate BertScore")
bertscore = load("bertscore")
# 'roberta-large_L17_no-idf_version=0.3.12(hug_trans=4.51.2)' 
bert0_results = bertscore.compute(predictions=[source], references=[target], lang="en")
# 'distilbert-base-uncased_L5_no-idf_version=0.3.12(hug_trans=4.51.2)'
bert1_results = bertscore.compute(predictions=[source], references=[target], model_type="distilbert-base-uncased")

# eval rouge 
#257 -> 2
#9 scores, what are src labels with oracle?

print("Evaluate ROUGE score")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
rouge_scores = scorer.score(source0, target0)
print(rouge_scores)

for src_str in source_l:
    for tar_str in target_l:
        scores = scorer.score(src_str, tar_str)
        print(scores)

from IPython import embed; embed()

## eval BLEU
## TODO need to cut into single word strings?
##print("Evaluate BLEU score")
##reference = [
##    'bb this is a dog'.split(),
##    'bb it is dog'.split(),
##    'bb dog it is'.split(),
##    'bb a dog, it is'.split()
##]
##print(reference)
##candidate = 'bb it is dog'.split()
##print('BLEU score -> {}'.format(sentence_bleu(reference, candidate))) # should be 1, not 0
#
#chencherry = SmoothingFunction()
#print(sentence_bleu(source, target, smoothing_function=chencherry.method1))
#print('BLEU score -> {}'.format(sentence_bleu(source, target))) # should be 1, not 0
#print('Individual 1-gram: %f' % sentence_bleu(source, target, weights=(1, 0, 0, 0)))
#print('Individual 2-gram: %f' % sentence_bleu(source, target, weights=(0, 1, 0, 0)))
#print('Individual 3-gram: %f' % sentence_bleu(source, target, weights=(0, 0, 1, 0)))
#print('Individual 4-gram: %f' % sentence_bleu(source, target, weights=(0, 0, 0, 1)))
#print(sentence_bleu(source, target, weights=(0.25, 0.25, 0.25, 0.25)))
#print(round(sentence_bleu([reference1, reference2, reference3], hypothesis2),4))
#print(sentence_bleu(source, target, weights=(1./5., 1./5., 1./5., 1./5., 1./5.))) 
#print(sentence_bleu(source, target, weights=[(1./2., 1./2.), (1./3., 1./3., 1./3.), (1./4., 1./4., 1./4., 1./4.)])) 
