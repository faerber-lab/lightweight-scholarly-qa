import requests
import time

from typing import Dict,List



def format_authors(author_list: List[Dict[str, str]]) -> str:
    author_str_list = []
    for author in author_list:
        name = author['name']
        split_names = name.split(" ")
        if not split_names[0].endswith('.'):
            name = " ".join([f"{split_names[0][0]}.", *split_names[1:]])
        author_str_list.append(name)
    
    return ", ".join(author_str_list)


def format_venue(data) -> str:
    venue = data['venue']

    journal = data['journal']

    if journal is not None:    
        if 'name' in journal:
            venue = journal['name']
        
        if 'volume' in journal:
            venue += ", " + journal['volume']
        
        if 'pages' in journal:
            venue += ", " + journal['pages']
    
    venue += ", " + str(data['year'])
    
    return venue


def fetch_paper_from_semantic_scholar(title: str):
    rate = 1.0
    while True:
        time.sleep(rate)
        rsp = requests.get((f'https://api.semanticscholar.org/graph/v1/paper/search/match?query={title}&fields=title,authors,venue,year,url,journal'))
        if rsp.status_code == 404:
            return None
        elif rsp.status_code == 429:
            print("SemanticScholar: Got status code 429 (rate limit)")
            rate = rate * 2
        else:
            break
            
        
    rsp.raise_for_status()
    results = rsp.json()

    if len(results['data']) == 0:
        return None
    
    data = results['data'][0]
    #print(data)
    
    title = data['title']
    authors = format_authors(data['authors'])
    venue = format_venue(data)
    url = data['url']

    return {"authors": authors, "title": title, "venue": venue, "url": url}



if __name__ == "__main__":
    query = input()
    print(fetch_paper_from_semantic_scholar(query))