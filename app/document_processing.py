
import uuid

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def enrich_metadata(filename: str):
    return {
        "source": filename,
        "document_id": str(uuid.uuid4())
    }
