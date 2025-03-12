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
    work_ids = []
    for work in range(len(ret['results']['bindings'])):
        wid = ret['results']['bindings'][work]['sub']['value']
        work_ids = work_ids + [wid]

    return work_ids

def work_title(soa_id: str) -> str:
    ret = get_obj(soa_id, "http://purl.org/dc/terms/title")
    return ret['results']['bindings'][0]['obj']['value']

def auth_created_works_names(soa_id: str) -> list:
    work_ids = auth_created_works(soa_id)
    work_names = []
    for wid in work_ids:
        work_names = work_names + [work_title(wid)]

    return work_names

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
    work_titles = auth_created_works_names(soa_id)

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

def get_work_data_by_name(name: str):
    # ending page, publucation date, starting page, creator, title, cites
    pass

get_auth_data_by_name("Bernhard Vogginger")
