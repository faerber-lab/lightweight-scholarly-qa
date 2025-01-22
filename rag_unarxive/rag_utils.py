import os
import json
import re
import subprocess
from tqdm import tqdm
from typing import List, Dict
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def sorted_os_walk(path):
    for dirpath, dirnames, filenames in os.walk(path):
        # Sort directory names and file names
        dirnames.sort()  # Sort in-place
        filenames.sort()  # Sort in-place
        yield dirpath, dirnames, filenames


def load_markdown_files(directory: str, max_files: int = None) -> List[Document]:
    """Load all markdown files from a directory and its subdirectories."""
    if directory[-1] != '/':
        directory = directory + '/'
    total = len(sorted(subprocess.run(f'find {directory} -name *.md' .split(), capture_output=True,text = True).stdout.strip().split()))
    if max_files is not None:
        total = min(total, max_files)
    print(f"Total documents: {total}")
    progressbar = tqdm(total=total)
    documents = []
    count=0
    for root, _, files in sorted_os_walk(directory):
        for file in files:
            if file.endswith('.md'):
                with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                    text = f.read()
                    documents.append(Document(page_content=text, metadata={"source": file}))
                count += 1
                progressbar.update(1)
                if count == max_files:
                    return documents
    return documents

def is_list_item(text: str) -> bool:
    """Check if text is a list item (bulleted or numbered)."""
    return bool(re.match(r'^[\s-]*[-\*\+]|^\s*\d+\.', text))

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into semantically meaningful chunks."""
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    # Custom text splitter that preserves semantic structure
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased for better context
        chunk_overlap=200,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    
    split_docs = []
    for doc in tqdm(documents):
        # First split by headers
        header_splits = markdown_splitter.split_text(doc.page_content)

        authors = None
        published_in = None
        doi = None
        abstract = None
        
        for header_split in header_splits:
            # Get the header information
            title, header1, section_title = (header_split.metadata.get("Header 1", ""),
                                       header_split.metadata.get("Header 2", ""),
                                       header_split.metadata.get("Header 3", ""))
            #print(title, header1, section_title)
            
            if header1 == "Authors":
                authors = header_split.page_content
                continue
            elif header1 == "Published in":
                published_in = header_split.page_content
                continue
            elif header1 == "DOI":
                doi = header_split.page_content
                continue
            elif header1 == "Abstract":
                abstract = header_split.page_content
                # here we do not do continue, we want to actually index the abstract!
            
            # Split content into lines
            lines = header_split.page_content.split('\n')
            current_chunk = []
            current_length = 0
            
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # Handle key-value pairs
                if ':' in line and not is_list_item(line):
                    current_chunk.append(line)
                    current_length += len(line)
                
                # Handle list items
                elif is_list_item(line):
                    list_items = [line]
                    # Collect all items in the current list
                    while i + 1 < len(lines) and (is_list_item(lines[i + 1].strip()) or not lines[i + 1].strip()):
                        i += 1
                        if lines[i].strip():
                            list_items.append(lines[i].strip())
                    current_chunk.extend(list_items)
                    current_length += sum(len(item) for item in list_items)
                
                # Regular text
                else:
                    current_chunk.append(line)
                    current_length += len(line)
                
                # Create new chunk if size limit reached
                if current_length >= 800:  # Slightly lower than chunk_size to account for overlap
                    current_chunk_text = '\n'.join(current_chunk)
                    split_docs.append(
                        Document(
                            page_content=f"Title: {title}\nAuthors: {authors}\nPublished in: {published_in}\nSection Title: {section_title}\n{current_chunk_text}",
                            metadata={
                                "source": doc.metadata["source"],
                                "title": title,
                                "section_title": section_title,
                                "authors": authors,
                                "published_in": published_in,
                                "doi": doi
                                # "abstract": abstract,  # we leave out the abstract to save some space. is this a good idea?
                            }
                        )
                    )
                    current_chunk = []
                    current_length = 0
                
                i += 1
            
            # Add remaining content as a chunk
            if current_chunk:
                current_chunk_text = '\n'.join(current_chunk)
                split_docs.append(
                    Document(
                        page_content=f"Title: {title}\nAuthors: {authors}\nPublished in: {published_in}\nSection Title: {section_title}\n{current_chunk_text}",
                        metadata={
                            "source": doc.metadata["source"],
                            "title": title,
                            "section_title": section_title,
                            "authors": authors,
                            "published_in": published_in,
                            "doi": doi
                            # "abstract": abstract,  # we leave out the abstract to save some space. is this a good idea?
                        }
                    )
                )
    
    return split_docs

def create_vector_store(documents: List[Document], embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2", faiss_index_file: str = "faiss_index", load=True, store=False, docs_per_batch=None) -> FAISS:
    """Create a FAISS vector store from documents."""
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, show_progress=True if docs_per_batch is None else False)
    if load:
        print("Loading the vector store from file")
        vector_store = FAISS.load_local(faiss_index_file, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Generating the vector store from scratch")
        if docs_per_batch == None:
            vector_store = FAISS.from_documents(documents, embeddings)
        else:            
            def batched(documents: List[Document], batch_size: int) -> List[Document]:    
                for i in range(0, len(documents), batch_size): 
                    yield documents[i:i+batch_size]

            with tqdm(total=len(documents)) as progress_bar:
                vector_store = None
                for docs in batched(documents, docs_per_batch):
                    if vector_store is None:
                        vector_store = FAISS.from_documents(docs, embeddings)
                    else:
                        vector_store.add_documents(docs)
                    progress_bar.update(len(docs))
                
        print("Storing the vector store to file")
        if store: 
            vector_store.save_local(faiss_index_file)
    return vector_store

def load_json_qa(json_path: str) -> List[Document]:
    """Load QA pairs from JSON and convert to documents."""
    with open(json_path, 'r') as f:
        qa_data = json.load(f)
    
    documents = []
    current_topic = None
    current_pairs = []
    
    for item in qa_data:
        if "conversations" in item:
            q = item["conversations"][0]["value"]
            a = item["conversations"][1]["value"]
            
            # Try to identify topic changes based on question content
            topic = q.split()[0] if q.split() else ""
            
            # If topic changed or accumulated enough pairs, create a new document
            if topic != current_topic or len(current_pairs) >= 3:
                if current_pairs:
                    content = "\n\n".join(current_pairs)
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": f"QA_{current_topic}",
                            "type": "qa",
                            "topic": current_topic
                        }
                    ))
                current_pairs = []
                current_topic = topic
            
            current_pairs.append(f"Q: {q}\nA: {a}")
    
    # Add remaining pairs
    if current_pairs:
        content = "\n\n".join(current_pairs)
        documents.append(Document(
            page_content=content,
            metadata={
                "source": f"QA_{current_topic}",
                "type": "qa",
                "topic": current_topic
            }
        ))
    
    return documents

def initialize_rag(markdown_dir: str, json_paths: List[str] = None, faiss_index_file: str = "faiss_index", load_index_from_file=True, store_index_to_file=False, max_files=None, docs_per_batch=None) -> FAISS:
    """Initialize RAG by loading both markdown and JSON QA documents."""

    if not load_index_from_file:
        # Load markdown documents
        print("Loading Markdown files...")
        documents = load_markdown_files(markdown_dir, max_files=max_files)
        print("\nSplitting them...")
        split_docs = split_documents(documents)
        
        # Load JSON QA documents if provided
        print("\nLoading JSON files...")
        if json_paths:
            for json_path in json_paths:
                qa_docs = load_json_qa(json_path)
                # No need to split QA docs as they're already in small chunks
                split_docs.extend(qa_docs)
    else:
        split_docs = []
    
    # Create vector store
    print("\nCreating Vector Store...")
    vector_store = create_vector_store(split_docs, faiss_index_file=faiss_index_file, load=load_index_from_file, store=store_index_to_file, docs_per_batch=docs_per_batch)
    
    return vector_store
