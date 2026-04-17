"""
rag.py
------
RAG pipeline: load documents → chunk → embed → store in ChromaDB → retrieve.
Supports .txt, .pdf, and .docx files from the configured docs/ folder.
"""

from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import (
    DOCS_FOLDER, CHROMA_FOLDER, EMBED_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, RAG_TOP_K, RAG_SEED_QUERIES,
)


# ── Document loaders ──────────────────────────────────────────────────────────

def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        print(f"  ⚠️  pypdf not installed — skipping {path.name}")
        return ""


def _load_docx(path: Path) -> str:
    try:
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        print(f"  ⚠️  python-docx not installed — skipping {path.name}")
        return ""


LOADERS = {".txt": _load_txt, ".pdf": _load_pdf, ".docx": _load_docx}


def load_documents(folder: Path = DOCS_FOLDER) -> list[dict]:
    """Load all supported docs from folder. Returns list of {text, source}."""
    if not folder.exists():
        print(f"  ℹ️  docs/ folder not found — RAG context will be empty.")
        return []

    docs = []
    for path in sorted(folder.iterdir()):
        loader = LOADERS.get(path.suffix.lower())
        if not loader:
            continue
        print(f"  📄 Loading {path.name}…")
        text = loader(path).strip()
        if text:
            docs.append({"text": text, "source": path.name})

    print(f"  ✅ Loaded {len(docs)} document(s)")
    return docs


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end].strip())
        start += size - overlap
    return [c for c in chunks if len(c) > 40]


# ── Vector store ─────────────────────────────────────────────────────────────

def build_vector_store(docs: list[dict]) -> tuple[chromadb.Collection, SentenceTransformer]:
    """Embed all document chunks and upsert into ChromaDB. Returns (collection, model)."""
    client = chromadb.PersistentClient(
        path=str(CHROMA_FOLDER),
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name="restaurant_docs",
        metadata={"hnsw:space": "cosine"},
    )

    # Determine which sources are already indexed
    existing_sources: set[str] = set()
    if collection.count() > 0:
        results = collection.get(include=["metadatas"])
        existing_sources = {m["source"] for m in results["metadatas"]}

    model = SentenceTransformer(EMBED_MODEL)
    new_chunks, new_ids, new_metas = [], [], []

    for doc in docs:
        if doc["source"] in existing_sources:
            print(f"  ♻️  {doc['source']} already indexed — skipping")
            continue
        for i, chunk in enumerate(chunk_text(doc["text"])):
            new_chunks.append(chunk)
            new_ids.append(f"{doc['source']}::{i}")
            new_metas.append({"source": doc["source"], "chunk": i})

    if new_chunks:
        print(f"  🔢 Embedding {len(new_chunks)} new chunks…")
        embeddings = model.encode(new_chunks, show_progress_bar=False).tolist()
        collection.add(
            documents=new_chunks,
            embeddings=embeddings,
            ids=new_ids,
            metadatas=new_metas,
        )
        print(f"  ✅ Vector store updated ({collection.count()} total chunks)")

    return collection, model


def retrieve_context(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    queries: list[str] = RAG_SEED_QUERIES,
    top_k: int = RAG_TOP_K,
) -> str:
    """
    Query the vector store with multiple seed queries.
    Returns deduplicated top chunks as a single context string.
    """
    if collection.count() == 0:
        return ""

    seen, chunks = set(), []
    for query in queries:
        embedding = model.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas"],
        )
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            key = f"{meta['source']}::{meta['chunk']}"
            if key not in seen:
                seen.add(key)
                chunks.append(f"[{meta['source']}]\n{doc}")

    return "\n\n".join(chunks)


def setup_rag() -> str:
    """
    Full RAG pipeline entry point.
    Load docs → embed → retrieve → return context string for system prompt.
    """
    print("\n📚 Setting up RAG knowledge base…")
    docs = load_documents()
    if not docs:
        return ""

    collection, model = build_vector_store(docs)
    context = retrieve_context(collection, model)

    if context:
        print(f"  ✅ RAG context ready ({len(context):,} chars injected into prompt)\n")
    return context
