#! /usr/bin/env python3

"""
SemOpenAlex SPARQL access
"""

from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper(
    "https://semopenalex.org/sparql"
)
sparql.setReturnFormat(JSON)

prefix = """
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT * WHERE {
"""

def get_obj(sub: str, pred: str, limit=1):
    sparql.setQuery(prefix + """
          <%s> <%s> ?obj .
        } LIMIT %s
        """ % (sub, pred, str(limit))
    )
    return sparql.queryAndConvert()

def get_sub(pred: str, obj: str, obj_is_id=True, limit=1):
    if obj_is_id: obj = "<"+obj+">"
    else:         obj = "\""+obj+"\""
    sparql.setQuery(prefix + """
          ?sub <%s> %s .
        } LIMIT %s
        """ % (pred, obj, str(limit))
    )
    return sparql.queryAndConvert()

def auth_soa_id(name: str) -> str:
    ret = get_sub("http://xmlns.com/foaf/0.1/name", name, False)
    return ret['results']['bindings'][0]['sub']['value']

def auth_name(soa_id: str) -> str:
    ret = get_obj(soa_id, "http://xmlns.com/foaf/0.1/name")
    return ret['results']['bindings'][0]['obj']['value']

def auth_num_works(soa_id: str) -> int:
    ret = get_obj(soa_id, "https://semopenalex.org/ontology/worksCount")
    return int(ret['results']['bindings'][0]['obj']['value'])

def auth_h_index(soa_id: str) -> int:
    ret = get_obj(soa_id, "http://purl.org/spar/bido/h-index")
    return int(ret['results']['bindings'][0]['obj']['value'])

def auth_cited_by_count(soa_id: str) -> int:
    ret = get_obj(soa_id, "https://semopenalex.org/ontology/citedByCount")
    return int(ret['results']['bindings'][0]['obj']['value'])

def auth_i10_index(soa_id: str) -> int:
    ret = get_obj(soa_id, "https://semopenalex.org/ontology/i10Index")
    return int(ret['results']['bindings'][0]['obj']['value'])

def auth_orcid_id(soa_id: str) -> str:
    ret = get_obj(soa_id, "https://dbpedia.org/ontology/orcidId")
    return ret['results']['bindings'][0]['obj']['value']

def auth_institute_id(soa_id: str) -> str:
    ret = get_obj(soa_id, "http://www.w3.org/ns/org#memberOf")
    return ret['results']['bindings'][0]['obj']['value']

def auth_institute_name(soa_id: str) -> str:
    institue_soa_id = auth_institute_id(soa_id)
    ret = get_obj(institue_soa_id, "http://xmlns.com/foaf/0.1/name")
    return ret['results']['bindings'][0]['obj']['value']

def auth_created_works(soa_id: str) -> list:
    ret = get_sub("http://purl.org/dc/terms/creator", soa_id, limit=1000)
    return get_ids_from_ret(ret)

def translate_ids(ids : list, func) -> list:
    translates = []
    for sid in ids:
        # use try, as sometimes infos are not available for individual items
        try:
            translates = translates + [func(sid)]
        except:
            pass
    return translates 

def auth_is_author_of(soa_id: str) -> list:
    # get work over authorship class, which includes author (+position), work and previous affiliation
    ret = get_sub("https://semopenalex.org/ontology/hasAuthor", soa_id, limit=1000)
    return ret['results']['bindings'][0]['sub']['value']

def get_type(soa_id: str) -> str:
    ret = get_obj(soa_id, "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", soa_id)
    return ret['results']['bindings'][0]['obj']['value']


# if function takes too long -> it's possible to bundle all info into one query
def get_auth_data_by_name(name: str):
    soa_id = auth_soa_id(name)
    num_works = auth_num_works(soa_id)
    h_index = auth_h_index(soa_id)
    cited_by_count = auth_cited_by_count(soa_id)
    i10_index = auth_i10_index(soa_id)
    orcid_id = auth_orcid_id(soa_id)
    institute = auth_institute_name(soa_id)
    #work_titles = auth_created_works_names(soa_id)
    work_titles = translate_ids(auth_created_works(soa_id), work_title)

    print("""
    %s
    -------------------
        num_works    = %s
        h_index      = %s
        citedByCount = %s
        i10Index     = %s
        orcidId      = %s
        institute    = %s
    -------------------
        works
    """ % (
        name,
        num_works,
        h_index,
        cited_by_count,
        i10_index,
        orcid_id,
        institute,
    ))
    for idx, work_str in enumerate(work_titles):
        print("            {}) {}".format(idx, work_str))


def work_soa_id(title: str) -> str:
    ret = get_sub("http://purl.org/dc/terms/title", title, False)
    return ret['results']['bindings'][0]['sub']['value']

def work_title(soa_id: str) -> str:
    ret = get_obj(soa_id, "http://purl.org/dc/terms/title")
    return ret['results']['bindings'][0]['obj']['value']

def work_doi(soa_id: str) -> str:
    ret = get_obj(soa_id, "http://purl.org/spar/datacite/doi")
    return ret['results']['bindings'][0]['obj']['value']

def work_type(soa_id: str) -> str:
    ret = get_obj(soa_id, "https://semopenalex.org/ontology/workType")
    return ret['results']['bindings'][0]['obj']['value']

def work_publication_date(soa_id: str) -> str:
    ret = get_obj(soa_id, "http://prismstandard.org/namespaces/basic/2.0/publicationDate")
    return ret['results']['bindings'][0]['obj']['value']

def work_cites(soa_id: str) -> str:
    ret = get_obj(soa_id, "https://semopenalex.org/ontology/citedByCount")
    return ret['results']['bindings'][0]['obj']['value']

def work_primary_topic(soa_id: str) -> str:
    ret = get_obj(soa_id, "https://semopenalex.org/ontology/hasPrimaryTopic")
    return ret['results']['bindings'][0]['obj']['value']

def work_abstract(soa_id: str) -> str:
    ret = get_obj(soa_id, "http://purl.org/dc/terms/abstract")
    return ret['results']['bindings'][0]['obj']['value']

def get_ids_from_ret(ret, sparql_type='sub') -> list:
    ids = []
    for entity in range(len(ret['results']['bindings'])):
        sid = ret['results']['bindings'][entity][sparql_type]['value']
        ids = ids + [sid]
    return ids

def work_creators(soa_id: str) -> str:
    ret = get_obj(soa_id, "http://purl.org/dc/terms/creator", limit=1000)
    return get_ids_from_ret(ret, 'obj')

def work_cited_works(soa_id: str) -> list:
    ret = get_obj(soa_id, "http://purl.org/spar/cito/cites", limit=1000)
    return get_ids_from_ret(ret, 'obj')

def work_cited_by_works(soa_id: str) -> list:
    ret = get_sub("http://purl.org/spar/cito/cites", soa_id, limit=1000)
    return get_ids_from_ret(ret)

def topic_name(soa_id: str) -> str:
    ret = get_obj(soa_id, "http://www.w3.org/2004/02/skos/core#prefLabel")
    return ret['results']['bindings'][0]['obj']['value']

def get_work_data_by_name(title: str):
    # ending page, publucation date, starting page, creator, title, cites
    soa_id = work_soa_id(title)
    title = work_title(soa_id)
    doi = work_doi(soa_id)
    wtype = work_type(soa_id)
    publication_date = work_publication_date(soa_id)
    cites = work_cites(soa_id)
    primary_topic = topic_name(work_primary_topic(soa_id))
    abstract = work_abstract(soa_id)
    authors = translate_ids(work_creators(soa_id), auth_name) # currently no order, use hasAuthorship instead
    cited_papers = work_cited_works(soa_id)
    cited_papers = translate_ids(work_cited_works(soa_id), work_title)
    cited_by_papers = translate_ids(work_cited_by_works(soa_id), work_title)

    print("""
    %s
    -------------------
        doi              = %s
        type             = %s
        publication_date = %s
        cites            = %s
        primary_topic    = %s
    -------------------
        abstract
        %s
    """ % (
        title,
        doi,
        wtype,
        publication_date,
        cites,
        primary_topic,
        abstract
    ))
    
    print("""
    -------------------
        authors 
    """)
    for idx, auth_str in enumerate(authors):
        print("            {}) {}".format(idx, auth_str))

    print("""
    -------------------
        cited_papers
    """)
    for idx, work_str in enumerate(cited_papers):
        print("            {}) {}".format(idx, work_str))

    print("""
    -------------------
        cited_by_papers
    """)
    for idx, work_str in enumerate(cited_by_papers):
        print("            {}) {}".format(idx, work_str))

get_work_data_by_name("Dynamic Power Management for Neuromorphic Many-Core Systems")
get_auth_data_by_name("Bernhard Vogginger")
