# Agentic AI eBook RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with LangGraph, ChromaDB, and Groq (Llama 3.3 70B) that answers questions strictly grounded in the Agentic AI eBook (Konverge AI).

## Architecture Overview

**Flow:** PDF → Chunking → Embeddings → ChromaDB → LangGraph (Retrieve → Rerank → Generate) → FastAPI → Streamlit

1. **Ingestion (`app/ingest.py`)**
   - Extracts text page-by-page using `pdfplumber`
   - Splits text into sections using heading-pattern detection (regex matching numbered headings like "2.3 Defining Characteristics of an Agent")
   - Merges consecutive sections sharing the same heading (handles repeated page-header titles)
   - Splits oversized sections (>700 chars) using `RecursiveCharacterTextSplitter` with 100-char overlap to avoid cutting ideas mid-sentence
   - Filters out low-signal chunks (<40 chars)
   - Generates embeddings using `all-MiniLM-L6-v2` (local, free, 384-dim)
   - Stores chunks + embeddings + metadata (page, heading) in a persistent ChromaDB collection

2. **Retrieval (`app/graph.py`)**
   - Query is embedded with the same MiniLM model
   - Top-8 candidates retrieved from ChromaDB via vector similarity
   - Re-ranked using `cross-encoder/ms-marco-MiniLM-L-6-v2`, keeping top-4 by true relevance (not just embedding proximity)
   - Confidence score = top cross-encoder relevance score

3. **Generation (`app/graph.py`)**
   - LangGraph with two nodes: `retrieve` → `generate`
   - Strict grounding system prompt: LLM must answer only from retrieved context, with an explicit refusal line for out-of-scope questions
   - LLM: Groq Llama 3.3 70B, temperature=0 for deterministic, factual answers

4. **API (`app/main.py`)** — FastAPI `POST /chat` endpoint, returns answer + confidence + source chunks as JSON

5. **UI (`app/ui.py`)** — Streamlit chat interface that calls the FastAPI endpoint over HTTP (not a direct import), keeping the API as the real, independently usable service

## Key Design Decisions

- **ChromaDB over Pinecone:** local, zero-setup, no external API dependency — removes a point of failure for reviewers running the project.
- **MiniLM embeddings over OpenAI:** free, local, no extra API key required beyond Groq.
- **Heading-based chunking over fixed-size chunking:** the eBook has a clean numbered heading structure (e.g. "2.3 Defining Characteristics..."), so splitting on headings keeps each concept (definition + example) intact instead of arbitrary character-count slicing.
- **Cross-encoder reranking:** the eBook has repetitive template phrasing across sections (e.g. "Example: The chatbot..."), which can fool pure cosine similarity. Reranking directly judges query-chunk relevance instead of embedding proximity, improving precision.
- **Known limitation:** a few PDF pages use a 2-column layout, which occasionally causes two headings to merge onto one extracted line. Documented and scoped as an acceptable trade-off for assignment timeframe rather than building custom column-detection logic.

## Setup Instructions

### Prerequisites
- Python 3.11+
- A free [Groq API key](https://console.groq.com)

### 1. Clone and set up environment
```bash
git clone https://github.com/divyankk14/agentic-ai-rag-chatbot.git
cd rag-chatbot

# Using uv (recommended)
uv sync

# OR using plain pip
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key
Copy `.env.example` to `.env` and add your Groq key:
