#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 s1304873 <s1304873@c7>
#
# Distributed under terms of the MIT license.

"""
"""

#Negatives: 
#    Please describe the paper FLARES VIII. The Emergence of Passive Galaxies in the Early Universe ($z > 5$) in a few sentences.
#    Condense the novel $VH+\text{jet}$ production in hadron-hadron collisions up to order $\alpha_s^3$ in perturbative QCD into a brief summary of its main themes and events.

# if NER fails -> can I fuzzy search for a paper title from the list of available papers?

import spacy
import random
import pickle
import numpy as np
from tqdm import tqdm
from spacy import util
from spacy.tokens import Doc
from spacy.training import Example
from spacy.language import Language
from classify_prompt import get_summarization_questions, get_simplification_questions
spacy.prefer_gpu()


def print_doc_entities(_doc: Doc):
    if _doc.ents:
        for _ent in _doc.ents:
            print(f"     {_ent.text} {_ent.label_}")
    else:
        print("     NONE")


def train_paper_ner(nlp: Language):
    # generate questions with paper titles as training/test data
    with open('title_to_file.pkl', 'rb') as f:
        title_to_file = pickle.load(f)

    list_of_works = list(title_to_file.keys())

    sum_questions, sum_doc_positions = get_summarization_questions(list_of_works, num_of_questions=200)
    sim_questions, sim_doc_positions = get_simplification_questions(list_of_works, num_of_questions=200)
    questions = np.stack((sum_questions, sim_questions))
    questions = np.concatenate((sum_questions, sim_questions))
    doc_positions = np.concatenate((sum_doc_positions, sim_doc_positions), axis=0)

    train_data = [None] * len(questions)
    for idx in range(len(questions)):
        train_data[idx] = (questions[idx], [(doc_positions[idx][0], doc_positions[idx][1], 'PAPER')])

    # Result before training
    print("Result before training:")
    print('--------------------------------------')
    print(sum_questions[0])
    print('--------------------------------------')
    doc = nlp(sum_questions[0])
    print('--------------------------------------')
    print_doc_entities(doc)

    # Disable all pipe components except 'ner'
    disabled_pipes = []
    for pipe_name in nlp.pipe_names:
        if pipe_name != 'ner':
            nlp.disable_pipes(pipe_name)
            disabled_pipes.append(pipe_name)

    print("Training ...")
    optimizer = nlp.create_optimizer()
    num_epochs = 10
    for epoch in range(num_epochs):
        random.shuffle(train_data)
        for raw_text, entity_offsets in tqdm(train_data, desc='{}/{}'.format(epoch+1, num_epochs)):
            doc = nlp.make_doc(raw_text)
            example = Example.from_dict(doc, {"entities": entity_offsets})
            spacy.training.offsets_to_biluo_tags(nlp.make_doc(raw_text), entity_offsets)
            nlp.update([example], sgd=optimizer)

    # Enable all previously disabled pipe components
    for pipe_name in disabled_pipes:
        nlp.enable_pipe(pipe_name)

    # Result after training
    sum_questions, sum_doc_positions = get_summarization_questions(list_of_works, num_of_questions=10)
    print("Result after training:")
    for q in sum_questions:
        doc = nlp(q)
        print('--------------------------------------')
        print(q)
        print('--------------------------------------')
        print_doc_entities(doc)
        print('--------------------------------------')

    # TODO check if author can be NER'd in text without/with paper in text
    ######
    ######
    ######
    ######
    ######

    # Save ner model
    #nlp.to_disk("/data/horse/ws/s1304873-llm_secrets/scholaryllm_prot/rag_unarxive/ner_nlp.spacy")

def main():
    nlp = spacy.load('en_core_web_sm')
    train_paper_ner(nlp)


if __name__ == '__main__':
    main()
