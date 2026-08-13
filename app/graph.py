
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from langgraph.graph import StateGraph, END

load_dotenv() 

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0  # 0 = deterministic, factual answers, no creative variation - important for grounding
)
 
class RAGState(TypedDict):
    question: str              
    retrieved_chunks: list
    confidence: float     
    answer: str           


# not inside functions , avoid reloading on every call
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("agentic_ai_ebook")


# RETRIEVE 
def retrieve_node(state: RAGState) -> dict:
    """
    Takes the question from state, does vector search + reranking,
    and returns updates to the state (retrieved_chunks, confidence).
    """
    query = state["question"]

    # Step A: vector search, wide net (k=8)
    query_embedding = embed_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=8)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Step B: rerank with cross-encoder, keep top 4
    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)

    combined = list(zip(documents, metadatas, scores))
    combined.sort(key=lambda x: x[2], reverse=True)
    top_chunks = combined[:4]

    # Step C: build clean chunk list + confidence score (top cross-encoder score)
    retrieved_chunks = [
        {"text": text, "page": meta["page"], "heading": meta["heading"], "score": float(score)}
        for text, meta, score in top_chunks
    ]
    confidence = float(top_chunks[0][2]) if top_chunks else 0.0

    return {
        "retrieved_chunks": retrieved_chunks,
        "confidence": confidence
    }



def generate_node(state: RAGState) -> dict:
    """
    Takes the retrieved chunks and question, builds a strict grounding prompt,
    calls the LLM, and returns the answer.
    """
    question = state["question"]
    chunks = state["retrieved_chunks"]

    # Combine chunk texts into one context block, labeled by page for traceability
    context = "\n\n".join(
        f"[Page {c['page']}] {c['text']}" for c in chunks
    )

    # This system prompt is the core of "strictly grounded" - it explicitly
    # forbids the LLM from using outside knowledge, and gives a required fallback line
    system_prompt = """You are a helpful assistant that answers questions using ONLY the provided context from the Agentic AI eBook.

Rules:
- Answer using ONLY information found in the context below.
- Do NOT use any outside knowledge, even if you know the answer.
- If the context does not contain enough information to answer the question, respond exactly with:
  "I don't have enough information in the provided document to answer this."
- Keep answers clear and concise.
- You may reference page numbers if helpful."""

    user_prompt = f"""Context:
{context}

Question: {question}

Answer based strictly on the context above:"""

    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    return {"answer": response.content}


# ---- BUILD THE FULL GRAPH ----
def build_graph():
    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("retrieve")   # graph starts here
    workflow.add_edge("retrieve", "generate")  # after retrieve, go to generate
    workflow.add_edge("generate", END)          # after generate, graph ends

    return workflow.compile()


# ---- BUILD THE GRAPH
if __name__ == "__main__":
    graph = build_graph()

    test_question_2 = "What is the capital of France?"
    result2 = graph.invoke({
        "question": test_question_2,
        "retrieved_chunks": [],
        "confidence": 0.0,
        "answer": ""
    })
    print(f"\n\nQuestion: {test_question_2}")
    print(f"Answer:\n{result2['answer']}")