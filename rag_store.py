# rag_store.py

import io
import uuid
from typing import List, Dict, Tuple

import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb

# Initialize embedding model and Chroma client once
_EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
_CHROMA_CLIENT = chromadb.Client()
_COLLECTION = _CHROMA_CLIENT.get_or_create_collection("documents_rag")


def pdf_to_text(pdf_bytes: bytes) -> str:
    """
    Extract raw text from a PDF using PyPDF2.
    """
    reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Simple sliding window chunking for long text.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Compute embeddings for a list of texts using sentence-transformers.
    """
    return _EMBED_MODEL.encode(texts, show_progress_bar=False).tolist()


def index_documents(files: List, namespace: str = "default") -> int:
    """
    Index a list of uploaded files (PDF/TXT/MD) into Chroma.
    Returns number of chunks added.
    """
    all_chunks: List[str] = []
    metadatas: List[Dict] = []
    ids: List[str] = []

    for f in files:
        name = f.name
        raw_bytes = f.read()

        if name.lower().endswith(".pdf"):
            try:
                text = pdf_to_text(raw_bytes)
            except Exception:
                text = ""
        else:
            text = raw_bytes.decode("utf-8", errors="ignore")

        if not text.strip():
            continue

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            ids.append(str(uuid.uuid4()))
            metadatas.append(
                {
                    "filename": name,
                    "chunk_index": idx,
                    "namespace": namespace,
                }
            )

    if not all_chunks:
        return 0

    embeddings = embed_texts(all_chunks)

    _COLLECTION.add(
        ids=ids,
        metadatas=metadatas,
        documents=all_chunks,
        embeddings=embeddings,
    )

    return len(all_chunks)


def query_rag(question: str, top_k: int = 5, namespace: str = "default") -> Tuple[str, List[Dict]]:
    """
    Query the vector store and return:
    - a combined context string with top-k chunks
    - a list of metadata dicts (filename, preview).
    """
    q_emb = embed_texts([question])[0]

    results = _COLLECTION.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"namespace": namespace},
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    context_parts = []
    sources: List[Dict] = []

    for doc, meta in zip(docs, metas):
        filename = meta.get("filename", "unknown")
        idx = meta.get("chunk_index", 0)
        context_parts.append(f"[FILE: {filename}, CHUNK {idx}]\n{doc}\n")
        sources.append(
            {
                "filename": filename,
                "chunk_index": idx,
                "preview": doc[:200],
            }
        )

    context = "\n\n--- DOCUMENT CHUNK ---\n\n".join(context_parts)
    return context, sources
