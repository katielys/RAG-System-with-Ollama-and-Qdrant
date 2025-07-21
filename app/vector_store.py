from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from app.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, OLLAMA_HOST, OLLAMA_MODEL
import ollama
import uuid

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
ollama_client = ollama.Client(host=OLLAMA_HOST)

def init_collection(vector_size: int = 4096):
    if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

def embed_text(text: str):
    response = ollama_client.embeddings(model=OLLAMA_MODEL, prompt=text)
    return response["embedding"]

def upsert_chunks(chunks, metadata):
    points = []
    for chunk in chunks:
        embedding = embed_text(chunk)
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk, **metadata}
        )
        points.append(point)
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
