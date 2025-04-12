#!/usr/bin/env python
# coding: utf-8

print("Loading argparse, sys, re...")
import argparse
import pprint
import os
import sys
import os.path
import nltk
from nltk.tokenize import sent_tokenize
from enum import Enum
from train_paper_ner_model import get_text_from_paper_title, search_for_paper_title
from references import References, clean_references, remove_after_excessive_linebreak, remove_cites_after_linebreak_or_dot, remove_generated_references
from thefuzz import fuzz
from references import References, clean_references
from classify_prompt import Task, classify_prompt

print("Loading llama_requests...")
from llama_request import llama_request

print("Loading others...")
from typing import Any, List, Tuple
print("Loading done...")

enable_fetch_version = True 

def multi_qa_context(references: References, no_rag: bool=False) -> str:
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
              "numbers should be added at the end of the sentences."

    if no_rag:
        context += "\n"
    else:
        context += "If no references  "\
              "are given, do not give an answer, instead point out that you could "\
              "not find any references!\nReferences:\n"
        context += "\nReferences:\n"
        context += references.format_for_context()

        #print(references.format_for_context())
        #pprint.pp(references)

    context += "Question: "

    return context

def single_qa_context(task: Task, gold_context: Any, references: References, no_rag: bool=False, zero_shot=True) -> str:
    """Format retrieved documents into a context string for the SingleQA task."""
    task_instructions = {
        "claim_no_context": ("Given a scientific claim, answer if the scientific claim is factually correct (true) or not (false). For each scientific claim provided, simply state whether it is true or false. If the statement is supported by the paragraph, answer true; otherwise answer false. You don't need to provide any explanation, just the label.", "\nClaim: "), 
        "claim_gold": ("Given a scientific claim and a gold paragraph that may support or contradict with the claim, answer if the scientific claim is factually correct or not. For each scientific claim provided, simply state whether it is true or false. If the statement is supported by the paragraph, answer true; otherwise answer false. You don't need to provide any explanation, just the label.", "\nClaim: "),
        "claim_full": ("Given a scientific claim and a set of relevant paragraphs, that may support or contradict with the claim, answer if the scientific claim is factually correct or not. For each scientific claim provided, simply state whether it is true or false. If the statement is supported by the paragraph, answer true; otherwise answer false. You don't need to provide any explanation, just the label. You also need to provide the citation numbers that support your answer. Your citation is presented as [i], where i corresponds to the number in the 'References: '.", "\nClaim: "),
        "boolean_question_no_context": ("Given a question related to scientific literature, answer yes or no. Simply state whether it is yes or no. You don't need to provide any explanation, just the label.", "\nQuestion: "),
        "boolean_question_gold": ("Given a question related to scientific literature and a gold paragraph that provides sufficient information to answer the question, answer yes or no. Simply state whether it is yes or no.","\nQuestion:" ),
        "boolean_question_full": ("Given a question related to scientific literature and a set of reference passages that may provide sufficient information to answer the question, answer yes or no. Simply state whether it is yes or no. You don't need to provide any explanation, just the label. You also need to provide the citation numbers that support your answer. Your citation is presented as [i], where i corresponds to the number in the 'References: '.", "\nQuestion: "),
        "yesnomaybe_question_no_context": ("Given a question related to scientific literature, answer yes, no or maybe. Simply state whether it is yes, no or maybe. You don't need to provide any explanation, just the label.", "\nQuestion: "),
        "yesnomaybe_question_gold": ("Given a question related to scientific literature and a gold paragraph that provides sufficient information to answer the question, answer yes, no or maybe. Simply state whether it is yes, no or maybe.","\nQuestion:" ),
        "yesnomaybe_question_full": ("Given a question related to scientific literature and a set of reference passages that may provide sufficient information to answer the question, answer yes, no or maybe. Simply state whether it is yes, no or maybe. You don't need to provide any explanation, just the label. You also need to provide the citation numbers that support your answer. Your citation is presented as [i], where i corresponds to the number in the 'References: '.", "\nQuestion: "),
    }

    demonstrations = {
        "claim_no_context": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this:\nClaim: 1 in 5 million in UK have abnormal PrP positivity.\n[Response_Start]false[Response_End]\nNow please verify the following claim.", 
        "claim_gold": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: \nReferences: \n[0] Title: Prevalent abnormal prion protein in human appendixes after bovine spongiform encephalopathy epizootic: large scale survey Text: OBJECTIVES To carry out a further survey of archived appendix samples to understand better the differences between existing estimates of the prevalence of subclinical infection with prions after the bovine spongiform encephalopathy epizootic and to see whether a broader birth cohort was affected, and to understand better the implications for the management of blood and blood products and for the handling of surgical instruments. DESIGN Irreversibly unlinked and anonymised large scale survey of archived appendix samples. SETTING Archived appendix samples from the pathology departments of 41 UK hospitals participating in the earlier survey, and additional hospitals in regions with lower levels of participation in that survey. SAMPLE 32,441 archived appendix samples fixed in formalin and embedded in paraffin and tested for the presence of abnormal prion protein (PrP). RESULTS Of the 32,441 appendix samples 16 were positive for abnormal PrP, indicating an overall prevalence of 493 per million population (95% confidence interval 282 to 801 per million). The prevalence in those born in 1941-60 (733 per million, 269 to 1596 per million) did not differ significantly from those born between 1961 and 1985 (412 per million, 198 to 758 per million) and was similar in both sexes and across the three broad geographical areas sampled. Genetic testing of the positive specimens for the genotype at PRNP codon 129 revealed a high proportion that were valine homozygous compared with the frequency in the normal population, and in stark contrast with confirmed clinical cases of vCJD, all of which were methionine homozygous at PRNP codon 129. CONCLUSIONS This study corroborates previous studies and suggests a high prevalence of infection with abnormal PrP, indicating vCJD carrier status in the population compared with the 177 vCJD cases to date. These findings have important implications for the management of blood and blood products and for the handling of surgical instruments.\nClaim: 1 in 5 million in UK have abnormal PrP positivity. \n[Response_Start]false[Response_End]\nNow please verify the following claim.\n",
        "claim_full": """
        Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: 
        References: 
        [0] Title: MLQA: Evaluating Cross-lingual Extractive Question Answering Text: Question answering (QA) models have shown rapid progress enabled by the availability of large, high-quality benchmark datasets. Such annotated datasets are difficult and costly to collect, and rarely exist in languages other than English, making building QA systems that work well in other languages challenging. In order to develop such systems, it is crucial to invest in high quality multilingual evaluation benchmarks to measure progress. We present MLQA, a multi-way aligned extractive QA evaluation benchmark intended to spur research in this area. MLQA contains QA instances in 7 languages, English, Arabic, German, Spanish, Hindi, Vietnamese and Simplified Chinese. MLQA has over 12K instances in English and 5K in each other language, with each instance parallel between 4 languages on average. We evaluate state-of-the-art cross-lingual models and machine-translation-based baselines on MLQA. In all cases, transfer results are shown to be significantly behind training-language performance.
        [1] Title: XOR QA: Cross-lingual Open-Retrieval Question Answering Text: Multilingual question answering tasks typically assume that answers exist in the same language as the question. Yet in practice, many languages face both information scarcity—where languages have few reference articles—and information asymmetry—where questions reference concepts from other cultures. This work extends open-retrieval question answering to a cross-lingual setting enabling questions from one language to be answered via answer content from another language. We construct a large-scale dataset built on 40K information-seeking questions across 7 diverse non-English languages that TyDi QA could not find same-language answers for. Based on this dataset, we introduce a task framework, called Cross-lingual Open-Retrieval Question Answering (XOR QA), that consists of three new tasks involving cross-lingual document retrieval from multilingual and English resources. We establish baselines with state-of-the-art machine translation systems and cross-lingual pretrained models. Experimental results suggest that XOR QA is a challenging task that will facilitate the development of novel techniques for multilingual question answering.
        [2] Title: Unsupervised Cross-lingual Representation Learning at Scale Text: This paper shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of cross-lingual transfer tasks. We train a Transformer-based masked language model on one hundred languages, using more than two terabytes of filtered CommonCrawl data. Our model, dubbed XLM-R, significantly outperforms multilingual BERT (mBERT) on a variety of cross-lingual benchmarks, including +14.6% average accuracy on XNLI, +13% average F1 score on MLQA, and +2.4% F1 score on NER. XLM-R performs particularly well on low-resource languages, improving 15.7% in XNLI accuracy for Swahili and 11.4% for Urdu over previous XLM models. We also present a detailed empirical analysis of the key factors that are required to achieve these gains, including the trade-offs between (1) positive transfer and capacity dilution and (2) the performance of high and low resource languages at scale. Finally, we show, for the first time, the possibility of multilingual modeling without sacrificing per-language performance; XLM-R is very competitive with strong monolingual models on the GLUE and XNLI benchmarks. We will make our code, data and models publicly available.
        Claim: The XOR QA dataset covers eight languages. 
        [Response_Start]false [1][Response_End]
        Now please verify the following claim.\n 
        """,
        "boolean_question_no_context": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this:\nQuestion: Did Chile's traffic law reform push police enforcement?\n[Response_Start]yes[Response_End]\nNow answer the following question.",
        "boolean_question_gold": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: \nReferences: \n[0] The objective of the current study is to determine to what extent the reduction of Chile's traffic fatalities and injuries during 2000-2012 was related to the police traffic enforcement increment registered after the introduction of its 2005 traffic law reform. A unique dataset with assembled information from public institutions and analyses based on ordinary least square and robust random effects models was carried out. Dependent variables were traffic fatality and severe injury rates per population and vehicle fleet. Independent variables were: (1) presence of new national traffic law; (2) police officers per population; (3) number of traffic tickets per police officer; and (4) interaction effect of number of traffic tickets per police officer with traffic law reform. Oil prices, alcohol consumption, proportion of male population 15-24 years old, unemployment, road infrastructure investment, years' effects and regions' effects represented control variables. Empirical estimates from instrumental variables suggest that the enactment of the traffic law reform in interaction with number of traffic tickets per police officer is significantly associated with a decrease of 8% in traffic fatalities and 7% in severe injuries. Piecewise regression model results for the 2007-2012 period suggest that police traffic enforcement reduced traffic fatalities by 59% and severe injuries by 37%. \nQuestion: Did Chile's traffic law reform push police enforcement?\n[Response_Start]yes[Response_End]\nNow answer the following question. ",
        "boolean_question_full": """
        Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: 
        References: 
        [0] The gap between evidence-based treatments and routine care has been well established. Findings from the Sequenced Treatments Alternatives to Relieve Depression (STAR*D) emphasized the importance of measurement-based care for the treatment of depression as a key ingredient for achieving response and remission; yet measurement-based care approaches are not commonly used in clinical practice. The Nine-Item Patient Health Questionnaire (PHQ-9) for monitoring depression severity was introduced in 19 diverse psychiatric practices. During the one-year course of the project the helpfulness and feasibility of implementation of PHQ-9 in these psychiatric practices were studied. The project was modeled after the Institute for Healthcare Improvement Breakthrough Series. Two of the 19 practices dropped out during the course of the project. By the conclusion of the study, all remaining 17 practices had adopted PHQ-9 as a routine part of depression care in their practice. On the basis of responses from 17 psychiatrists from those practices, PHQ-9 scores influenced clinical decision making for 93% of 6,096 patient contacts. With the additional information gained from the PHQ-9 score, one or more treatment changes occurred during 40% of these clinical contacts. Changing the dosage of antidepressant medication and adding another medication were the most common treatment changes recorded by psychiatrists, followed by starting or increasing psychotherapy and by switching or initiating antidepressants. In 3% of the patient contacts, using the PHQ-9 led to additional suicide risk assessment. 
        [1] To compare maternal and neonatal outcomes among grandmultiparous women to those of multiparous women 30 years or older. A database of the vast majority of maternal and newborn hospital discharge records linked to birth/death certificates was queried to obtain information on all multiparous women with a singleton delivery in the state of California from January 1, 1997 through December 31, 1998. Maternal and neonatal pregnancy outcomes of grandmultiparous women were compared to multiparous women who were 30 years or older at the time of their last birth. The study population included 25,512 grandmultiparous and 265,060 multiparous women 30 years or older as controls. Grandmultiparous women were predominantly Hispanic (56%). After controlling for potential confounding factors, grandmultiparous women were at significantly higher risk for abruptio placentae (odds ratio OR: 1.3; 95% confidence intervals CI: 1.2-1.5), preterm delivery (OR: 1.3; 95% CI: 1.2-1.4), fetal macrosomia (OR: 1.5; 95% CI: 1.4-1.6), neonatal death (OR: 1.5; 95% CI: 1.3-1.8), postpartum hemorrhage (OR: 1.2; 95% CI: 1.1-1.3) and blood transfusion (OR: 1.5; 95% CI: 1.3-1.8).', 'long_answer': 'Grandmultiparous women had increased maternal and neonatal morbidity, and neonatal mortality even after controlling for confounders, suggesting a need for closer observation than regular multiparous patients during labor and delivery.
        [2] The objective of the current study is to determine to what extent the reduction of Chile's traffic fatalities and injuries during 2000-2012 was related to the police traffic enforcement increment registered after the introduction of its 2005 traffic law reform. A unique dataset with assembled information from public institutions and analyses based on ordinary least square and robust random effects models was carried out. Dependent variables were traffic fatality and severe injury rates per population and vehicle fleet. Independent variables were: (1) presence of new national traffic law; (2) police officers per population; (3) number of traffic tickets per police officer; and (4) interaction effect of number of traffic tickets per police officer with traffic law reform. Oil prices, alcohol consumption, proportion of male population 15-24 years old, unemployment, road infrastructure investment, years' effects and regions' effects represented control variables. Empirical estimates from instrumental variables suggest that the enactment of the traffic law reform in interaction with number of traffic tickets per police officer is significantly associated with a decrease of 8% in traffic fatalities and 7% in severe injuries. Piecewise regression model results for the 2007-2012 period suggest that police traffic enforcement reduced traffic fatalities by 59% and severe injuries by 37%. 
        Question: Did Chile's traffic law reform push police enforcement?
        [Response_Start]yes [2][Response_End]
        Now answer the following question. 
        """,
        "yesnomaybe_question_no_context": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this:\nQuestion: Did Chile's traffic law reform push police enforcement?\n[Response_Start]yes[Response_End]\nNow answer the following question.",
        "yesnomaybe_question_gold": "Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: \nReferences: \n[0] The objective of the current study is to determine to what extent the reduction of Chile's traffic fatalities and injuries during 2000-2012 was related to the police traffic enforcement increment registered after the introduction of its 2005 traffic law reform. A unique dataset with assembled information from public institutions and analyses based on ordinary least square and robust random effects models was carried out. Dependent variables were traffic fatality and severe injury rates per population and vehicle fleet. Independent variables were: (1) presence of new national traffic law; (2) police officers per population; (3) number of traffic tickets per police officer; and (4) interaction effect of number of traffic tickets per police officer with traffic law reform. Oil prices, alcohol consumption, proportion of male population 15-24 years old, unemployment, road infrastructure investment, years' effects and regions' effects represented control variables. Empirical estimates from instrumental variables suggest that the enactment of the traffic law reform in interaction with number of traffic tickets per police officer is significantly associated with a decrease of 8% in traffic fatalities and 7% in severe injuries. Piecewise regression model results for the 2007-2012 period suggest that police traffic enforcement reduced traffic fatalities by 59% and severe injuries by 37%. \nQuestion: Did Chile's traffic law reform push police enforcement?\n[Response_Start]yes[Response_End]\nNow answer the following question. ",
        "yesnomaybe_question_full": """
        Your answer must be marked by special tokens, [Response_Start] and [Response_End]. For example, the input and output looks like this: 
        References: 
        [0] The gap between evidence-based treatments and routine care has been well established. Findings from the Sequenced Treatments Alternatives to Relieve Depression (STAR*D) emphasized the importance of measurement-based care for the treatment of depression as a key ingredient for achieving response and remission; yet measurement-based care approaches are not commonly used in clinical practice. The Nine-Item Patient Health Questionnaire (PHQ-9) for monitoring depression severity was introduced in 19 diverse psychiatric practices. During the one-year course of the project the helpfulness and feasibility of implementation of PHQ-9 in these psychiatric practices were studied. The project was modeled after the Institute for Healthcare Improvement Breakthrough Series. Two of the 19 practices dropped out during the course of the project. By the conclusion of the study, all remaining 17 practices had adopted PHQ-9 as a routine part of depression care in their practice. On the basis of responses from 17 psychiatrists from those practices, PHQ-9 scores influenced clinical decision making for 93% of 6,096 patient contacts. With the additional information gained from the PHQ-9 score, one or more treatment changes occurred during 40% of these clinical contacts. Changing the dosage of antidepressant medication and adding another medication were the most common treatment changes recorded by psychiatrists, followed by starting or increasing psychotherapy and by switching or initiating antidepressants. In 3% of the patient contacts, using the PHQ-9 led to additional suicide risk assessment. 
        [1] To compare maternal and neonatal outcomes among grandmultiparous women to those of multiparous women 30 years or older. A database of the vast majority of maternal and newborn hospital discharge records linked to birth/death certificates was queried to obtain information on all multiparous women with a singleton delivery in the state of California from January 1, 1997 through December 31, 1998. Maternal and neonatal pregnancy outcomes of grandmultiparous women were compared to multiparous women who were 30 years or older at the time of their last birth. The study population included 25,512 grandmultiparous and 265,060 multiparous women 30 years or older as controls. Grandmultiparous women were predominantly Hispanic (56%). After controlling for potential confounding factors, grandmultiparous women were at significantly higher risk for abruptio placentae (odds ratio OR: 1.3; 95% confidence intervals CI: 1.2-1.5), preterm delivery (OR: 1.3; 95% CI: 1.2-1.4), fetal macrosomia (OR: 1.5; 95% CI: 1.4-1.6), neonatal death (OR: 1.5; 95% CI: 1.3-1.8), postpartum hemorrhage (OR: 1.2; 95% CI: 1.1-1.3) and blood transfusion (OR: 1.5; 95% CI: 1.3-1.8).', 'long_answer': 'Grandmultiparous women had increased maternal and neonatal morbidity, and neonatal mortality even after controlling for confounders, suggesting a need for closer observation than regular multiparous patients during labor and delivery.
        [2] The objective of the current study is to determine to what extent the reduction of Chile's traffic fatalities and injuries during 2000-2012 was related to the police traffic enforcement increment registered after the introduction of its 2005 traffic law reform. A unique dataset with assembled information from public institutions and analyses based on ordinary least square and robust random effects models was carried out. Dependent variables were traffic fatality and severe injury rates per population and vehicle fleet. Independent variables were: (1) presence of new national traffic law; (2) police officers per population; (3) number of traffic tickets per police officer; and (4) interaction effect of number of traffic tickets per police officer with traffic law reform. Oil prices, alcohol consumption, proportion of male population 15-24 years old, unemployment, road infrastructure investment, years' effects and regions' effects represented control variables. Empirical estimates from instrumental variables suggest that the enactment of the traffic law reform in interaction with number of traffic tickets per police officer is significantly associated with a decrease of 8% in traffic fatalities and 7% in severe injuries. Piecewise regression model results for the 2007-2012 period suggest that police traffic enforcement reduced traffic fatalities by 59% and severe injuries by 37%. 
        Question: Did Chile's traffic law reform push police enforcement?
        [Response_Start]yes [2][Response_End]
        Now answer the following question. 
        """,
    }

    if no_rag:
        if gold_context is None:
            if task == Task.SINGLEQA_YESNO:
                taskname = "boolean_question_no_context"
            elif task == Task.SINGLEQA_YESNOMAYBE:
                taskname = "yesnomaybe_question_no_context"
            else:
                raise ValueError(f"Unknown Task!, got: {task=}")
            
            context = task_instructions[taskname][0]

            if not zero_shot:
                context += demonstrations[taskname]
            
            context += task_instructions[taskname][1]

        else:
            if task == Task.SINGLEQA_YESNO:
                taskname = "boolean_question_gold"
            elif task == Task.SINGLEQA_YESNOMAYBE:
                taskname = "yesnomaybe_question_gold"
            else:
                raise ValueError(f"Unknown Task!, got: {task=}")

            context = task_instructions[taskname][0]

            if not zero_shot:
                context += demonstrations[taskname]
            
            context += "\nReferences: \n"

            if isinstance(gold_context, dict):
                context += "[0] " + " ".join(gold_context.values())
            
            context += task_instructions[taskname][1]
    else:
        if task == Task.SINGLEQA_YESNO:
            taskname = "boolean_question_full"
        elif task == Task.SINGLEQA_YESNOMAYBE:
            taskname = "yesnomaybe_question_full"
        else:
            raise ValueError(f"Unknown Task!, got: {task=}")
        
        context = task_instructions[taskname][0]

        if not zero_shot:
            context += demonstrations[taskname]
    
        context += "\nReferences: \n"
        context += references.format_for_context()

        context += task_instructions[taskname][1]
        

    return context


def generate_response(prompt: str, task: Task, context: Any=None, initial: bool=False, previous_chat=None, references: References|None=None, no_rag: bool=False) -> Any:
    """Generate a response"""

    # If this is the first call, we potentially have to add system prompt or RAG context
    if initial:
        # Format context
        if task == Task.MULTIQA:
            # check if references were found
            if no_rag or (references and len(references) > 0):
                context = multi_qa_context(references, no_rag)
                # Combine context and prompt
                full_prompt = context + prompt
                messages = [
                    {"role": "user", "content": full_prompt}
                ]
            else:
                return {'generated_text': [{'role': 'assistant', 'content': "Unfortunately, no references related to your question were found!"}]}
        elif task in [Task.SINGLEQA_YESNO, Task.SINGLEQA_YESNOMAYBE]:
            # check if references were found
            #if no_rag: # or (references and len(references) > 0):
            context = single_qa_context(task=task, gold_context=context, references=references, no_rag=no_rag)
            # Combine context and prompt
            full_prompt = context + prompt
            messages = [
                {"role": "user", "content": full_prompt}
            ]
            #else:
            #    return {'generated_text': [{'role': 'assistant', 'content': "Unfortunately, no references related to your question were found!"}]}
        elif task == Task.SUMMARIZATION_SCITLDR:
            context = "You will be shown the text of the abstract, introduction, " \
                      "and conclusion of a scientific paper. Please summarize the " \
                      "key findings of the work in 1-2 sentences.\n\n"
            full_prompt = context + prompt
            messages = [
                {"role": "user", "content": full_prompt}
            ]

        elif task == Task.SIMPLIFICATION or task == Task.SUMMARIZATION:
            content = ""
            if enable_fetch_version:
                # get paper title w/ NER and do a fuzzy search in RAG database + add text to prompt + send to LLM 
                nltk.download('punkt')
                number_of_sentences = sent_tokenize(prompt)
                if number_of_sentences<3:
                    title = search_for_paper_title(prompt)
                    if title is not None:
                        content = get_text_from_paper_title(search_for_paper_title(prompt, fthresh=70))
                        context = "You will be shown the full text of 1 or 2 scientific papers. "
                        if task == Task.SIMPLIFICATION: 
                            context = context + "Please simplify the work in a simpler " \
                                                "language for easier understanding.\n\n"
                        elif task == Task.SUMMARIZATION:
                            context = context + "Please summarize the key findings of " \
                                                "the work in a few sentences.\n\n"

            full_prompt = content + prompt
            messages = [
                {"role": "user", "content": full_prompt}
            ]
        
        elif task == Task.FACT_REQUEST:
            # TODO
            #chat = generate_response_kg_request(prompt)
            pass
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

    chat = llama_request(messages, port=str(os.environ.get("LLAMA_PORT", 8000)))

    return chat


def chatbot(prompt: str|None, context=None, continuous_chat: bool=True, print_outputs=True, no_rag=False, no_clean_refs=False, rag_topk=10, remove_gen_reflist=True, no_remove_after_excessive_linebreak=False, no_remove_cite_after_linebreak_or_dot=False, fixed_task=Task.MULTIQA) -> None|Tuple[str, References, str]:
    #print("\nStarting ...")

    initial = True
    prompt_classification = True if fixed_task is None else False
    chat = None
    references = None

    global ner_nlp
    # commented out - already done in global scope
    #ner_nlp.from_disk("/data/horse/ws/s1304873-llm_secrets/scholaryllm_prot/rag_unarxive/ner_nlp.spacy")

    # make interactive rag
    while True:
        if prompt is None or not initial:
            prompt = input("Ask your question. type 'quit' to exit. \nYou: ")
        if prompt == "quit":
            break
        if not prompt:
            continue
        
        # Fixed task is set in default parameters - set to None if it should be classified
        if prompt_classification:
            print ("The task of the prompt is {}".format(classify_prompt(prompt)))
            prompt = None
            continue
        else:
            task = fixed_task

        print(f"{task=}")

        # Generate and stream response
        references = None
        if initial:
            references = References()

            if task == Task.MULTIQA:
                if no_rag:
                    references = References()
                else:
                    references = References.retrieve_from_vector_store(prompt, topk=rag_topk, port=int(os.environ.get("RAG_PORT", 8003)))
                    references.drop_refs_with_low_score(threshold=0.1)

                chat = generate_response(prompt, task=task, initial=initial, references=references, no_rag=no_rag)
            elif task in [Task.SINGLEQA_YESNO, Task.SINGLEQA_YESNOMAYBE]:
                if no_rag:
                    references = References()
                else:
                    references = References.retrieve_from_vector_store(prompt, topk=rag_topk, port=int(os.environ.get("RAG_PORT", 8003)))
                    references.drop_refs_with_low_score(threshold=0.1)
                chat = generate_response(prompt, task=task, context=context, initial=initial, references=references, no_rag=no_rag)
        else:
            chat = generate_response(prompt, task=Task.FOLLOWUPQUESTION, initial=initial, references=references, previous_chat=chat['generated_text'], no_rag=no_rag)

        reply = chat['generated_text'][-1]['content']

        if references:
            if not no_clean_refs:
                reply, references = clean_references(reply, references)
                references.update_from_semanticscholar()
            formatted_references = references.format_for_references()
        else:
            formatted_references = ""
        
        orig_reply = reply

        if remove_gen_reflist:
            reply = remove_generated_references(reply)
        
        if not no_remove_after_excessive_linebreak:
            reply = remove_after_excessive_linebreak(reply)
        
        if not no_remove_cite_after_linebreak_or_dot:
            reply = remove_cites_after_linebreak_or_dot(reply)

        #print(f"\n{remove_gen_reflist=}, {no_remove_after_excessive_linebreak=}, {no_remove_cite_after_linebreak_or_dot=}\n")

        chat['generated_text'][-1]['content'] = reply

        # Print the response (+ potentially References)
        if print_outputs:
            print("\nResponse:\n")
            print(reply)
            if len(formatted_references) > 0:
                print("\nReferences:\n")
                print(formatted_references)
            print("")
            print(orig_reply)
            print("")
        initial = False
        if not continuous_chat:
            return reply, references, orig_reply


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--no_clean_refs", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    chatbot(prompt=args.prompt, rag_topk=args.topk, no_clean_refs=args.no_clean_refs)



