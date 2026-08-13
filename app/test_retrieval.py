import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder


def query_chroma(query: str, k: int = 5):
    """
    Loads the existing chroma_db collection, embeds the query using the SAME
    model we used for ingestion, and returns the top-k most similar chunks.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_collection("agentic_ai_ebook")

    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return results


def rerank_results(query: str, results: dict, top_n: int = 4):
    """
    Takes Chroma's raw results, scores each (query, chunk) pair with a
    cross-encoder, and returns the top_n chunks sorted by true relevance
    instead of raw vector distance.
    """
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    documents = results["documents"][0]
    ids = results["ids"][0]
    metadatas = results["metadatas"][0]

    pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(pairs)

    combined = list(zip(ids, documents, metadatas, scores))
    combined.sort(key=lambda x: x[3], reverse=True)

    return combined[:top_n]


if __name__ == "__main__":
    test_query = "What is the role of memory in an agentic AI system?"

    results = query_chroma(test_query, k=8)

    print("=== BEFORE RERANKING (raw vector search) ===\n")
    for i in range(len(results["ids"][0])):
        print(f"{i+1}. {results['ids'][0][i]} | Distance: {results['distances'][0][i]:.4f} | {results['metadatas'][0][i]['heading']}")

    reranked = rerank_results(test_query, results, top_n=4)

    print("\n=== AFTER RERANKING (cross-encoder) ===\n")
    for i, (chunk_id, text, meta, score) in enumerate(reranked):
        print(f"{i+1}. {chunk_id} | Score: {score:.4f} | {meta['heading']}")
        print(f"   {text[:150]}...\n")