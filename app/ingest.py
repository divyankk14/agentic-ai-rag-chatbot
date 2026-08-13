import pdfplumber
import re 
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


PDF_PATH = r"data\Ebook-Agentic-AI.pdf"

# func to extra data from each page

def extract_text_by_page(pdf_path :str )->  list[dict]: # type hint that the pdf in string and return list of dictinories
    """
    Opens the PDF and extracts text from each page.
    Returns a list of dicts: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    """
    pages_data = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_data.append({
                    "page" : i + 1, # to amek the count start from 1
                    "text" : text
                })
                
    return pages_data


# function combine all the text from the pages  and split into section 
def split_into_sections(pages_data: list[dict]) ->list[dict]:
    
    
    # Removed ^...$ anchors so it can find a heading pattern ANYWHERE in a line,
    # not just at the very start - needed because 2-column PDF pages sometimes
    # glue two headings onto the same extracted line.
    heading_pattern = re.compile(r'(?m)^(\d\.\d+\s+[A-Z][^\n]{3,60})$')
    
    sections= []
    current_heading = "introduction"
    current_text = []
    current_page = pages_data[0]["page"] if pages_data else 1
    
    for page in pages_data:
        lines = page["text"].split("\n")
        for line in lines:
            match = heading_pattern.match(line.strip()) # does this line look like a heading?
            
            
            if match: # foound a NEW heading
                if current_text:
                    sections.append({
                        "heading":current_heading,
                        "text":"\n".join(current_text).strip(), # join lines into one gaint text 
                        "page":current_page
                    })
                current_heading = match.group(1).strip()
                current_text = []
                current_page = page["page"]
            else:
                current_text.append(line)
                
            
    if current_text :
        sections.append({
            'heading': current_heading,
            "text":"\n".join(current_text).strip(),
            "page": current_page
        })
        
    return sections
                



def merge_duplicate_sections(sections: list[dict]) -> list[dict]:
    """
    If consecutive sections share the same heading (e.g. a repeated page-header
    title like '1.5 Agentic AI Use cases' appearing on 3 pages), merge them
    into a single section instead of keeping them as separate fragments.
    """
    merged = []

    for sec in sections:
        if merged and merged[-1]["heading"] == sec["heading"]:
            # Same heading as the previous section .. merge text together
            merged[-1]["text"] += "\n" + sec["text"]
        else:
            # New heading .. start a fresh section
            merged.append(sec)

    return merged


def split_oversized_sections(sections:list[dict], max_chars: int= 700 , overlap:int = 100) -> list[dict]:
    """
    Any section text longer than max_chars gets further split using
    RecursiveCharacterTextSplitter, which tries to cut at paragraph/sentence
    boundaries instead of blind character counts. Small sections pass through unchanged.
    Each resulting chunk keeps the same heading/page metadata as its parent section.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = max_chars,
        chunk_overlap= overlap,
        separators=["\n\n", "\n", ". ", " "]  # trying these split points in order of preference
    )
    
    final_chunks = []
    
    for sec in sections:
        if len(sec["text"])<= max_chars:
            # Small enough already so we will keep as one chunk
            final_chunks.append({
                "heading" :sec["heading"],
                "page": sec["page"],
                "text": sec["text"]
            })
        else:
            sub_chunks=splitter.split_text(sec["text"])
            for sub in sub_chunks:
                final_chunks.append({
                "heading" :sec["heading"],
                "page": sec["page"],
                "text": sub
                })
    return final_chunks 



def filter_and_tag_chunks(chunks : list[dict], min_chars:int = 40) -> list[dict]:
    """
    Drops very short chunks (likely headers, noise)
    and assigns each remaining chunk a unique chunk_id for tracking through
    the pipeline (embedding, storage, retrieval debugging).
    """
    clean_chunks = []
    chunk_id = 0
    
    for c in chunks:
        text = c["text"].strip()
        if len(text) < min_chars:
            continue
        
        clean_chunks.append({
            "chunk_id":f"chunks_{chunk_id}",
            "heading":c["heading"],
            "page":c["page"],
            "text":text
        })
        
        chunk_id += 1
        
    return clean_chunks



def generate_embeddings(chunks: list[dict]) ->list[dict]:
    """
    Loads the MiniLM embedding model and generates a vector for each chunk's text
    Adds an embedding key to each chunk dict.
    """   
    model = SentenceTransformer("all-MiniLM-l6-v2")
    
    texts = [c["text"]for c in chunks ]
    embeddings = model.encode(texts , show_progress_bar=True)
    
    for c,emb in zip(chunks , embeddings):
        c["embedding"] = emb.tolist()
    
    return chunks 
        
    
    
    
def store_in_chromadb(chunks :list[dict],persist_dir :str = "chroma_db", collection_name: str = "agentic_ai_ebook"):
    """
    Stores each chunk's text, embedding, and metadata into a persistent ChromaDB collection.
    """
    
    client = chromadb.PersistentClient(path=persist_dir) 
    
    
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        
    collection = client.create_collection(name=collection_name)
    
    collection.add(
        ids=[c["chunk_id"] for c in chunks ], # unique id for each chuck
        embeddings= [c["embedding"] for c in chunks], #  vectors
        documents= [c['text'] for c in chunks ], # imp text 
        metadatas= [{"heading":c["heading"], "page": c["page"]} for c in chunks ] # extra info for debugging
    )
    
    print(f"stored {collection.count()} chunks in chromadb collection'{collection_name}' ")
    
    
    return collection


import chromadb
from sentence_transformers import SentenceTransformer

def query_chroma(query: str, k: int = 5):
    """
    Loads the existing chroma_db collection, embeds the query using the SAME
    model we used for ingestion, and returns the top-k most similar chunks.
    """
    # Load the same embedding model used during ingestion
    # IMPORTANT: must use the same model here as in ingest.py, otherwise
    # the query vector and stored vectors won't be comparable
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to the same persistent Chroma DB we built in Step 10
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("agentic_ai_ebook")

    # Embed the query the same way we embedded chunks
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return results


if __name__ == "__main__":
    test_query = "What is the role of memory in an agentic AI system?"

    results = query_chroma(test_query, k=5)

    print(f"Query: {test_query}\n")
    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]
        text = results["documents"][0][i]

        print(f"--- Result {i+1} ---")
        print(f"ID: {chunk_id} | Distance: {distance:.4f} | Page: {metadata['page']} | Heading: {metadata['heading']}")
        print(f"Text: {text[:200]}...\n")




# if __name__ == "__main__":
#     pages = extract_text_by_page(PDF_PATH)
#     print(f"Extracted text from {len(pages)} pages")

#     sections = split_into_sections(pages)
#     sections = merge_duplicate_sections(sections)
#     print(f"Merged into {len(sections)} sections")

#     chunks = split_oversized_sections(sections)
#     print(f"After size-splitting: {len(chunks)} chunks")

#     chunks = filter_and_tag_chunks(chunks)
#     print(f"After noise filtering: {len(chunks)} final chunks\n")
    
#     chunks = generate_embeddings(chunks)
#     print(f"Generated embeddings. Example vector length: {len(chunks[0]['embedding'])}\n")

#     for c in chunks[:3]:
#         print(f"{c['chunk_id']} | embedding[:5] = {c['embedding'][:5]}")
        
#     chunks = generate_embeddings(chunks)
#     print(f"Generated embeddings. Example vector length: {len(chunks[0]['embedding'])}")

#     store_in_chromadb(chunks)