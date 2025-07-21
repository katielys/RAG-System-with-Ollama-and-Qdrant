# Placeholder for hybrid retrieval and reranking
from app.vector_store import qdrant
from app.config import COLLECTION_NAME

def search_qdrant(query_vector, top_k=5):
    return qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
