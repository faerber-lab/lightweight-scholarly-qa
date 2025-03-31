import re

from dataclasses import dataclass
from typing import Union, List, Tuple, Iterable

import semanticscholar
import rag_request

@dataclass
class Reference:
    title: str
    authors: str
    venue: str
    url: str
    content: str
    score: float

    def update_from_semanticscholar(self) -> None:
        semanticscholar_result = semanticscholar.fetch_paper_from_semantic_scholar(self.title)
        res = semanticscholar_result
        # if we cannot find the reference on semanticscholar, we still use our own info which is usually worse,
        # but it is what we have...
        if res is not None:
            self.authors = res["authors"]
            self.title = res["title"]
            self.venue = res["venue"]
            self.url = res["url"]

    def format_for_context(self, id: int) -> str:
        content_cleaned = self.content
        content_cleaned.replace("{{cite:}}", "")
        content_cleaned.replace(", ,", "")
        content_cleaned.replace(" , ", ", ")
        content_cleaned.replace("..", "")
        content_cleaned.replace("\n", " ")
        content_cleaned.replace("  ", " ")
        return f"[{id}] {content_cleaned}\n"
    
    def format_for_references(self, id: int) -> str:
        reftext = f"[{id}] {self.authors}. \"{self.title}\". {self.venue}, {self.url}"
        reftext.replace(", ,", "")
        reftext.replace(" , ", ", ")
        reftext.replace("..", "")
        reftext.replace("\n", " ")
        reftext.replace("  ", " ")
        reftext = reftext + "\n"
        return reftext

    def cleaned_content(self) -> str:
        content_cleaned = self.content
        content_cleaned.replace("{{cite:}}", "")
        content_cleaned.replace(", ,", "")
        content_cleaned.replace(" , ", ", ")
        content_cleaned.replace("..", "")
        content_cleaned.replace("\n", " ")
        content_cleaned.replace("  ", " ")
        return f"{content_cleaned}"


class References(list):

    def __init__(self, *args: Reference|list[Reference]|tuple[Reference]) -> None:
        super().__init__()
        self.extend(args)  # Use extend to validate elements


    @classmethod
    def retrieve_from_vector_store(cls, prompt: str, topk: int=10, port: int=8003) -> "References":
        retrieved_docs = rag_request.rag_request(prompt, k=topk, port=port)
        refs = cls.from_retrieved_docs(retrieved_docs)
        return refs
    
    @classmethod
    def from_retrieved_docs(cls, retrieved_docs) -> "References":
        refs = References()
        for doc, score in retrieved_docs:
            if doc is not None:
                authors = doc["metadata"].get("authors", "unknown authors")
                title = doc["metadata"].get("title", "unknown title")
                venue = doc["metadata"].get("published_in", "")
                url = doc["metadata"].get("doi", "")
                content = doc["page_content"]
                content_split = content.split("\n")
                main_content = " ".join(content_split[4:])
                reference = Reference(title, authors, venue, url, main_content, score)
                refs.append(reference)
        return refs

    def update_from_semanticscholar(self) -> None:
        for ref in self:
            ref: Reference
            ref.update_from_semanticscholar()

    def format_for_context(self) -> str:
        return "".join([ref.format_for_context(i+1) if ref is not None else f"[{i+1}] None" for i,ref in enumerate(self)])

    def format_for_references(self) -> str:
        return "".join([ref.format_for_references(i+1) if ref is not None else f"[{i+1}] None" for i,ref in enumerate(self)])
    
    def drop_refs_with_low_score(self, threshold: float) -> None:
        """Removes all Reference objects with a score below the given threshold."""
        self[:] = [ref for ref in self if ref.score >= threshold]        

    def __getitem__(self, index: Union[int, slice]) -> Union[Reference, 'References']:
        result = super().__getitem__(index)
        if isinstance(index, slice):
            return References(*result)  # Return a new References instance for slices
        return result

    def __add__(self, other: object) -> 'References':
        if isinstance(other, References):
            return References(*self, *other)
        raise TypeError("Can only concatenate with another References object")

    def __iadd__(self, other: 'References') -> 'References':
        if isinstance(other, References):
            self.extend(other)
            return self
        raise TypeError("Can only extend with another References object")
    

def remove_unused_references(reply: str, references: References) -> Tuple[str, References]:
    for i,ref in enumerate(references):
        if not f"[{i+1}]" in reply:
            references[i] = None
    return reply, references

def uniquify_and_renumber_references(reply: str, references: References) -> Tuple[str, References]:
    """
    references contain "authors", "title", "venue" and "url"
    """
    replace = []
    previous_refs = {}
    running_idx = 1

    titles = [ref.title if ref is not None else None for ref in references]

    # find duplicate references, compute the new indices for the 
    # remaining unified references.
    for i, (title, ref) in enumerate(zip(titles, references)):
        prev_idx = i+1
        if title is None:
            continue
        if title in previous_refs:
            new_idx = previous_refs[title]["idx"]
            replace.append((prev_idx, new_idx))
        else:
            previous_refs[title] = {"idx": running_idx, "ref": ref}
            if running_idx != prev_idx:
                replace.append((prev_idx, running_idx))
            running_idx += 1
            
    # replace the old reference indices in the reply by the new ones
    for prev, repl in replace:
        reply = reply.replace(f"[{prev}]", f"[{repl}]")

    new_refs = []
    for title, val in previous_refs.items():
        new_refs.append(val['ref'])
    
    new_refs = References(*new_refs)
                
    return reply, new_refs
    
def clean_citations(text: str) -> str:
    def process_match(match):
        citations = re.findall(r'\d+', match.group())  # Extract numbers from the matched citations
        unique_sorted_citations = sorted(set(map(int, citations)))  # Remove duplicates and sort
        return ''.join(f'[{num}]' for num in unique_sorted_citations)  # Reconstruct sorted citations

    # Replace all citation groups with cleaned versions
    cleaned_text = re.sub(r'(\[(\d+)\])+', process_match, text)
    return cleaned_text

def remove_generated_references(text: str) -> str:
    split_text = text.split("References:")
    return split_text[0]

def clean_references(reply: str, references: References) -> Tuple[str, References]:
    reply, references = remove_unused_references(reply, references)
    reply, references = uniquify_and_renumber_references(reply, references)

    reply = clean_citations(reply)
    reply = remove_generated_references(reply)
    return reply, references
    

if __name__ == "__main__":
    refs = References.retrieve_from_vector_store("What is SpiNNaker2?")

    print(refs.format_for_context())
    print(refs.format_for_references())

    print("=======")

    print(refs[2])

    x = refs[0]
    y = refs[1:2]
    print(type(x), type(y))
