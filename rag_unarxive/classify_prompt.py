from llama_request import llama_request
from enum import Enum, StrEnum
import numpy as np
import random
from tqdm import tqdm

# chemsum_single_document_summarization
# summarization:
#       https://huggingface.co/datasets/abisee/cnn_dailymail
#       science.scitldr_aic: https://huggingface.co/datasets/allenai/scitldr
# simplification:
#       https://huggingface.co/datasets/MichaelR207/MultiSim

# true
# Please summarize the paper 'Attention is all you need'. -> Summarization
# What is the h index of Florian Kelber? -> Fact-Request

# false
# Please summarize the paper 'Attention is all you need' -> text, true if eval
# What is the structure of Spinnaker2? -> Simplification
# What is SpiNNaker2? -> Fact-Request/Simplification


SYSTEM_PROMPT_CLASSIFIER = """
You are a classifier who takes a user prompt and classify it into 3 different classes dependent on the question or
demand. The available classes are summarization, simplification and question-answering.
"""

SYSTEM_PROMPT_CLASSIFIER = """
You will be presented with a task text. Classify the intent behind this task by choosing from one of the following categories:
- Summarization: reduce a provided longer text to a smaller one by filtering important information
- Simplification: change a provided text to make it easier to understand
- Fact-Request: identify and retrieve specific requested metadata related to author or work
- Question-Answering: answer general questions

Your answer should be a single word from the following list of options: ["Summarization", "Simplification", "Fact-Request", "Question-Answering", ]. Do not include any other text in your response.
"""

SYSTEM_PROMPT_CLASSIFIER_EVAL = """
You will be presented with a classification text. Please reduce the text to a single word. Your answer should be a single word from the following list of options: ["Summarization", "Simplification", "Fact-Request", "Question-Answering", ]. Do not include any other text in your response.
"""

SYSTEM_PROMPT_CLASSIFIER_NO_KG = """
You will be presented with a task text. Classify the intent behind this task by choosing from one of the following categories:
- Summarization: reduce a provided longer text to a smaller one by filtering important information
- Simplification: change a provided text to make it easier to understand
- Question-Answering: answer a general question

Your answer should be a single word from the following list of options: ["Summarization", "Simplification", "Question-Answering"]. Do not include any other text in your response.
"""

KG_ACCESS_CLASSIFIER = """
You will be presented with a metadata request. Classify the intent behind this task by choosing from one of the following categories:

- number-of-works: get the number of works or papers from an Author
- cited-by-count: get how ofter an Author was cited
- institute: get the institute from an author
- works-or-papers: get all works or papers by name from an author
- type: get the work type 
- publication-date: get the work publication date
- cites: get how often a work was cited
- topic: get the primary topic of a work
- abstract: get the work abstract
- authors: get the list of authors from a work
- cited-papers: get the list of papers that were cited in work
- cited-by-papers: get the list of papers which cite the work
- other: everything else

Your answer should be a single word from the following list of options: ["number-of-works", "cited-by-count", "institute", "works-or-papers", "type", "publication-date", "cites", "topic", "abstract", "authors", "cited-papers", "cited-by-papers", "other"]. Do not include any other text in your response.
"""

KG_ACCESS_CLASSIFIER_AUTHOR = """
You will be presented with a metadata request for an author. Classify the intent behind this task by choosing from one of the following categories:

- number-of-works: get the number of works or papers from an author
- cited-by-count: get how ofter an author was cited
- institute: get the institute from an author
- works-or-papers: get all works or papers by name from an author
- other: everything else

Your answer should be a single word from the following list of options: ["number-of-works", "cited-by-count", "institute", "works-or-papers", "other"]. Do not include any other text in your response.
"""

KG_ACCESS_CLASSIFIER_WORK = """
You will be presented with a metadata request for a work or paper. Classify the intent behind this task by choosing from one of the following categories:

- type: get the work type 
- publication-date: get the work publication date
- cites: get how often a work was cited
- topic: get the primary topic of a work
- abstract: get the work abstract
- authors: get the list of authors from a work
- cited-papers: get the list of papers that were cited in work
- cited-by-papers: get the list of papers which cite the work
- other: everything else

Your answer should be a single word from the following list of options: ["type", "publication-date", "cites", "topic", "abstract", "authors", "cited-papers", "cited-by-papers", "other"]. Do not include any other text in your response.
"""


SOA_METADATA = [
    "author number of works",
    "author h-index",
    "author cited by count",
    "author i10-Index",
    "author orcid-Id",
    "author institute",
    "author works",
    "work doi",
    "work type",
    "work publication date",
    "work cites",
    "work topic",
    "work abstract",
    "work authors",
    "work cited papers",
    "work cited-by papers",
]


class Task(Enum):
    """
    Task class of a prompt. Simplification is more specific than summarization is more specific than question-answering.
    """
    MULTIQA = 0
    FOLLOWUPQUESTION = 1
    SIMPLIFICATION = 2 
    SUMMARIZATION = 3
    FACT_REQUEST = 4 
    UNSPECIFIED = 6
    #SIMPLIFICATION = "simplification"
    #SUMMARIZATION = "summarization"
    #FACT_REQUEST = "fact-request"
    #QUESTION_ANSWERING = "question-answering"
    #UNSPECIFIED = "unspecified"


class KGTemplate(Enum):
    """
    Knowledge-Graph Template class
    """
    AUTHOR_NUMBER_OF_WORKS = 0
    AUTHOR_HINDEX = 1
    AUTHOR_CITED_BY_COUNT = 2
    AUTHOR_I10INDEX = 3
    AUTHOR_ORCID = 4
    AUTHOR_INSTITUTE = 5
    AUTHOR_WORKS = 6
    WORK_DOI = 7
    WORK_TYPE = 8
    WORK_PUBLICATION_DATE = 9
    WORK_CITES = 10
    WORK_TOPIC = 11
    WORK_ABSTRACT = 12
    WORK_AUTHORS = 13
    WORK_CITED_PAPERS = 14
    WORK_CITEDBY_PAPERS = 15


def get_summarization_questions(list_of_works, num_of_questions=10):
    questions = np.array([
        "Please summarize the paper [[doc]].",
        "Please describe the paper [[doc]] in a few sentences.",
        "Please shorten the paper [[doc]].",
        "Please break down the paper [[doc]] into short notes.",
        "Summarize the key points of the article [[doc]] in one paragraph.",
        "Condense the novel [[doc]] into a brief summary of its main themes and events.",
        "Provide a bullet-point summary of [[doc]].",
        "Break down [[doc]] into key takeaways.",
        "Can you turn [[doc]] into a list of concise notes?",
        "Extract the main points from [[doc]] and format them into a summary.",
        "Summarize the contents of [[doc]].",
        "Create a short executive summary of [[doc]].",
        "Condense [[doc]] into three sentences.",
        "What are the key ideas from [[doc]]?",
    ])

    #gen_questions = np.empty((num_of_questions), dtype=np.dtypes.StringDType)
    gen_questions = np.empty((num_of_questions), dtype=np.dtypes.StrDType)
    doc_positions = np.empty((num_of_questions, 2), dtype=int)
    print("Generate {} summarization questions".format(num_of_questions))
    for gen_idx in tqdm(range(num_of_questions)):
        q_idx = random.randint(0, len(questions)-1)
        w_idx = random.randint(0, len(list_of_works)-1)
        doc_start = questions[q_idx].find('[[doc]]')
        doc_positions[gen_idx] = [doc_start, doc_start+len(list_of_works[w_idx])]
        gen_questions[gen_idx] = questions[q_idx].replace('[[doc]]', list_of_works[w_idx])

        ## test if start/end position are borders of tokens
        ##nlp.tokenizer.explain(gen_questions[gen_idx])
        ##from IPython import embed; embed()
        #doc = nlp(gen_questions[gen_idx])
        #fb_ent = doc.char_span(doc_start, doc_start+len(list_of_works[w_idx]), label="PAPER")
        #print(type(fb_ent))

    return gen_questions, doc_positions


def get_simplification_questions(list_of_works, num_of_questions=10):
    questions = np.array([
        "Rewrite the research findings on [[doc]] [[grade]].",
        "Break down the complex terms in [[doc]] [[grade]].",
        "Simplify the paper [[doc]] [[grade]]. ",
        "Explain [[doc]] [[grade]].",
        "Simplify [[doc]] [[grade]].",
        "Rephrase [[doc]] [[grade]].",
        "Rewrite [[doc]] [[grade]].",
        "Please explain the paper [[doc]] [[grade]]. ",
        "Please simplify [[doc]] [[grade]].",
        "Can you please explain the paper [[doc]] [[grade]]? ",
        "Can you simplify [[doc]] [[grade]]?",
        "Can you make [[doc]] understandable [[grade]]?",
        "Can you break down [[doc]] [[grade]]?",
        "Can you simplify [[doc]] [[grade]]?",
        "Can you describe [[doc]] [[grade]]?",
        #"[[grad]] and can't understand the paper [[doc]]. Please explain.\n "+
    ])

    grades = np.array([
        ""
        "in plain, everyday language",
        "in simple terms",
        "in a way that a high school student could understand.",
        "in a way that's easy to grasp, step by step",
        "in a way that's easy for first-time readers",
        "in a way that would make sense to someone who’s never taken computer science",
        "for a highschool student",
        "for a child",
        "for a non-medical audience",
        "for someone without a background in science",
        "for the general public",
        "so that someone without a computer science background can understand it",
        "so that it’s accessible to someone with no prior knowledge of computer science",
        "into a few easy-to-follow steps",
        "into a shorter, clearer sentence",
        "into three straightforward steps for a beginner",
        "into easy-to-follow points",
        "using a metaphor or analogy that’s simple to understand",
        "like I am 5",
    ])

    simplification_grades_start = np.array([
        ""
        "I am an undergrad",
        "I am a highschool student",
        "I am 5",
        "I am not well versed in computer science",
    ])

    #gen_questions = np.empty((num_of_questions), dtype=np.dtypes.StringDType)
    gen_questions = np.empty((num_of_questions), dtype=np.dtypes.StrDType)
    doc_positions = np.empty((num_of_questions, 2), dtype=int)
    print("Generate {} simplification questions".format(num_of_questions))
    for gen_idx in tqdm(range(num_of_questions)):
        q_idx = random.randint(0, len(questions)-1)
        w_idx = random.randint(0, len(list_of_works)-1)
        g_idx = random.randint(0, len(grades)-1)
        doc_start = questions[q_idx].find('[[doc]]')
        doc_positions[gen_idx] = [doc_start, doc_start+len(list_of_works[w_idx])]
        doc_repl               = questions[q_idx].replace('[[doc]]', list_of_works[w_idx])
        gen_questions[gen_idx] = doc_repl.replace('[[grade]]', grades[g_idx])

        ## test if start/end position are borders of tokens
        ##nlp.tokenizer.explain(gen_questions[gen_idx])
        ##from IPython import embed; embed()
        #doc = nlp(gen_questions[gen_idx])
        #fb_ent = doc.char_span(doc_start, doc_start+len(list_of_works[w_idx]), label="PAPER")
        #print(type(fb_ent))

    return gen_questions, doc_positions


def classify_prompt(prompt: str, num_council: int=1):
    """Classify the prompt into the categories: simplification, summarization, question-answering, kg-request"""

    prompt_task = prompt

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_CLASSIFIER
        },
        {"role": "user", "content": prompt_task}
    ]

    messages_reduce = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_CLASSIFIER_EVAL
        },
        {"role": "user", "content": prompt_task}
    ]

    messages_kg_eval = [
        {
            "role": "system",
            "content": KG_ACCESS_CLASSIFIER
        },
        {"role": "user", "content": prompt_task}
    ]

    messages_kg_eval_author = [
        {
            "role": "system",
            "content": KG_ACCESS_CLASSIFIER_AUTHOR
        },
        {"role": "user", "content": prompt_task}
    ]

    messages_kg_eval_work = [
        {
            "role": "system",
            "content": KG_ACCESS_CLASSIFIER_WORK
        },
        {"role": "user", "content": prompt_task}
    ]

    this_class = Task.UNSPECIFIED
    classes = []
    #for member in range(num_council):
    chat = llama_request(messages, port=8000)
    answer = chat['generated_text'][2]['content']
    # try to condense to one word if generation ignored rule
    for reduce_text_try in range(10):
        #if " " in answer:
        if answer in ['Summarization', 'Simplification', 'Fact-Request', 'Question-Answering']:
            break
        else:
            print("need to reduce answer -> try #{}".format(reduce_text_try))
            messages_reduce[1]['content'] = answer
            chat = llama_request(messages_reduce, port=8000)
            answer = chat['generated_text'][2]['content']

    def findWholeWord(w):
        return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

    if   answer == 'Summarization':
        return Task.SUMMARIZATION 
    elif answer == 'Simplification':
        return Task.SIMPLIFICATION 
    elif answer == 'Fact-Request':
        return Task.FACT_REQUEST 
    elif answer == 'Question-Answering':
        return Task.MULTIQA 

    return Task.UNSPECIFIED

