from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from app import document_processing, vector_store, retriever, llm_integration, evaluation

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "RAG system with Qdrant is running"}

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    chunks = document_processing.chunk_text(text)
    metadata = document_processing.enrich_metadata(file.filename)
    vector_store.init_collection()
    # You can add logic here to insert chunks into Qdrant
    return {"message": f"Processed {len(chunks)} chunks from {file.filename}"}

class QueryRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 5

@app.post("/query/")
def query_chunks(request: QueryRequest):
    results = retriever.search_qdrant(request.query_vector, request.top_k)
    return {"results": results}

class AnswerRequest(BaseModel):
    query: str
    context: str

@app.post("/answer/")
def generate_answer(request: AnswerRequest):
    result = llm_integration.generate_answer(request.query, request.context)
    return {"answer": result["answer"], "score": result["score"]}

class EvaluateRequest(BaseModel):
    predicted: str
    ground_truth: str

@app.post("/evaluate/")
def evaluate_answer(request: EvaluateRequest):
    score = evaluation.exact_match(request.predicted, request.ground_truth)
    return {"exact_match": score}
