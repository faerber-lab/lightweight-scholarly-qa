#! /usr/bin/env python3

import soa_sparql_access as soa
from train_paper_ner_model import get_text_from_paper_title, search_for_paper_title, search_for_author_name
"""

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


def identify_kg_request(prompt):
    # check KG request by identifying specific word
    use_spacy = False
    if   any(findWholeWord(x)(answer.lower()) for x in  ['h-index', 'hindex', 'h index', 'h_index']):
        kg_task = KGTemplate.AUTHOR_HINDEX
    elif any(findWholeWord(x)(answer.lower()) for x in  ['i10-index', 'i10index', 'i10 index', 'i10_index']):
        kg_task = KGTemplate.AUTHOR_I10INDEX
    elif any(findWholeWord(x)(answer.lower()) for x in  ['orcid-id', 'orcidid', 'orcid id', 'orcid_id', 'orcid']):
        kg_task = KGTemplate.AUTHOR_ORCID
    elif any(findWholeWord(x)(answer.lower()) for x in  ['doi']):
        kg_task = KGTemplate.WORK_DOI
    else:
        # TODO
        # check if author or work in prompt
        has_author = False
        has_work = False

        title = search_for_paper_title(prompt)
        author = search_for_author_name(prompt)

        if title in not None:
            has_work = True 

        if author in not None:
            has_author = True 

        # TODO
        # check KG request with either llama or spacyTextCategorizer
        if use_spacy:
            pass
        else:
            if has_author and has_work:
                chat = llama_request(messages_kg_eval, port=8000)
                answer = chat['generated_text'][2]['content']

            elif has_author:
                entity = author
                chat = llama_request(messages_kg_eval_author, port=8000)
                answer = chat['generated_text'][2]['content']

            elif has_work:
                entity = work 
                chat = llama_request(messages_kg_eval_work, port=8000)
                answer = chat['generated_text'][2]['content']

            else:
                print('ERROR: Prompt identified as KG access, but no author or work found.')
                return None

    return kg_task, entity

def get_kg_response(kg_task, entity):
    if   kg_task = KGTemplate.AUTHOR_NUMBER_OF_WORKS:
        num_works = soa.auth_num_works(soa.auth_soa_id(entity))
        return str(num_works)
    elif kg_task = KGTemplate.AUTHOR_HINDEX:
        h_index = soa.auth_h_index(soa.auth_soa_id(entity))
        return str(h_index)
    elif kg_task = KGTemplate.AUTHOR_CITED_BY_COUNT:
        cited_by_count = soa.auth_cited_by_count(soa.auth_soa_id(entity))
        return str(cited_by_count)
    elif kg_task = KGTemplate.AUTHOR_I10INDEX:
        i10_index = soa.auth_i10_index(soa.auth_soa_id(entity))
        return str(i10_index)
    elif kg_task = KGTemplate.AUTHOR_ORCID:
        orcid_id = soa.auth_orcid_id(soa.auth_soa_id(entity))
        return str(orcid_id)
    elif kg_task = KGTemplate.AUTHOR_INSTITUTE:
        institute = soa.auth_institute_name(soa.auth_soa_id(entity))
        return str(institute)
    elif kg_task = KGTemplate.AUTHOR_WORKS:
        work_titles = soa.translate_ids(soa.auth_created_works(soa.auth_soa_id(entity)), work_title)
        return(" ".join(work_titles))
    elif kg_task = KGTemplate.WORK_DOI:
        doi = soa.work_doi(soa.work_soa_id(entity))
        return str(doi)
    elif kg_task = KGTemplate.WORK_TYPE:
        wtype = soa.work_type(soa.work_soa_id(entity))
        return str(wtype)
    elif kg_task = KGTemplate.WORK_PUBLICATION_DATE:
        publication_date = soa.work_publication_date(soa.work_soa_id(entity))
        return str(publication_date)
    elif kg_task = KGTemplate.WORK_CITES:
        cites = soa.work_cites(soa.work_soa_id(entity))
        return str(cites)
    elif kg_task = KGTemplate.WORK_TOPIC:
        primary_topic = topic_name(soa.work_primary_topic(soa.work_soa_id(entity)))
        return str(primary_topic)
    elif kg_task = KGTemplate.WORK_ABSTRACT:
        abstract = soa.work_abstract(soa.work_soa_id(entity))
        return str(abstract)
    elif kg_task = KGTemplate.WORK_AUTHORS:
        authors = soa.translate_ids(soa.work_creators(soa.work_soa_id(entity)), soa.auth_name) # currently no order, use hasAuthorship instead
        return(" ".join(authors))
    elif kg_task = KGTemplate.WORK_CITED_PAPERS:
        cited_papers = soa.translate_ids(soa.work_cited_soa.works(soa.work_soa_id(entity)), soa.work_title)
        return(" ".join(cited_papers))
    elif kg_task = KGTemplate.WORK_CITEDBY_PAPERS:
        cited_by_papers = soa.translate_ids(soa.work_cited_by_soa.works(soa.work_soa_id(entity)), soa.work_title)
        return(" ".join(cited_by_papers))
    else
        return "Apologies, but I could not find out what Knowledge-Graph-Access you wanted."


def generate_response_kg_request(prompt : str):
    return get_kg_response(identify_kg_request(prompt))


if __name__ == "__main__":
    generate_response_kg_request("Give me the h index of Matthias Jobst")


