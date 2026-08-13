from fastapi import FastAPI
from pydantic import BaseModel
from app.graph import build_graph

# APP SETUP 
app = FastAPI(title="Agentic AI eBook RAG Chatbot")
graph = build_graph()


# REQUEST RESPONSE 
class ChatRequest(BaseModel):
    question: str  # what the client must send

class ChunkInfo(BaseModel):
    text: str
    page: int
    heading: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    sources: list[ChunkInfo]


# ENDPOINT 
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Takes a question, runs it through the RAG graph, returns the answer,
    confidence score, and the source chunks used to generate the answer.
    """
    result = graph.invoke({
        "question": request.question,
        "retrieved_chunks": [],
        "confidence": 0.0,
        "answer": ""
    })

    return ChatResponse(
        answer=result["answer"],
        confidence=result["confidence"],
        sources=result["retrieved_chunks"]
    )


#HEALTH CHECK 
@app.get("/")
def root():
    return {"status": "RAG chatbot API is running"}
