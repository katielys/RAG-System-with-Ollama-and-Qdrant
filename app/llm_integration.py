import ollama
from app.config import OLLAMA_HOST, OLLAMA_MODEL

ollama_client = ollama.Client(host=OLLAMA_HOST)

def generate_answer(query: str, context: str):
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = ollama_client.chat(model=OLLAMA_MODEL, messages=[
        {"role": "user", "content": prompt}
    ])
    return {"answer": response["message"]["content"], "score": 1.0}
