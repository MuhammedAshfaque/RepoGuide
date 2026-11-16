"""
Indexer for CodeCompass.

Orchestrates the full indexing pipeline:
  1. Load files from a repo (via loader)
  2. Chunk each file (via chunker)
  3. Embed chunks in small batches (via sentence-transformers)
  4. Store embeddings + metadata in ChromaDB (persistent on disk)

Also provides functions to check if an index already exists and to
load the embedding model (once, lazily) so it can be shared between
indexing and querying.

Memory notes (8 GB RAM):
  - The embedding model (all-MiniLM-L6-v2) is ~90 MB in memory.
  - We embed in batches of EMBEDDING_BATCH_SIZE (default 32) to avoid
    building a huge list of embeddings in memory at once.
  - ChromaDB runs in-process with persistent storage on disk.
"""

import hashlib
import json
import time
from pathlib import Path

from src.config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    get_collection_name,
)
from src.ingestion.loader import load_files
from src.ingestion.chunker import chunk_file, Chunk
from src.retrieval.keyword_search import build_bm25_index, get_bm25_index_path


# ── ChromaDB + Embedding Globals ─────────────────────────────────────────────
# These are loaded lazily (on first use) so importing this module is cheap.

_embedding_model = None
_chroma_client = None

CHROMA_DIR = DATA_DIR / "chroma_db"


def get_embedding_model():
    """
    Load the sentence-transformers embedding model (lazy singleton).

    Returns the model object. The first call downloads/loads the model
    (~90 MB); subsequent calls return the cached instance.
    """
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"[indexer] Loading embedding model: {EMBEDDING_MODEL} ...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"[indexer] Embedding model loaded.")
    return _embedding_model


def get_chroma_client():
    """
    Get or create a persistent ChromaDB client.

    Data is stored on disk at data/chroma_db/ so indexes survive
    between runs.
    """
    global _chroma_client
    if _chroma_client is None:
        import chromadb
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        print(f"[indexer] ChromaDB client ready (storage: {CHROMA_DIR})")
    return _chroma_client


_collection_name = get_collection_name


def _chunk_id(chunk: Chunk, index: int) -> str:
    """
    Generate a unique, deterministic ID for a chunk.

    Uses a hash of the file path + line range so re-indexing the same
    repo produces the same IDs (enabling idempotent upserts).
    """
    key = f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}:{index}"
    return hashlib.md5(key.encode()).hexdigest()


# ── Main Indexing Pipeline ───────────────────────────────────────────────────

def index_repo(
    source: str,
    force_reindex: bool = False,
    force_reclone: bool = False,
) -> dict:
    """
    Full indexing pipeline: load → chunk → embed → store in ChromaDB.

    Args:
        source:         GitHub URL or local directory path.
        force_reindex:  If True, delete existing ChromaDB collection and re-index.
        force_reclone:  If True, re-clone the repo even if it exists locally.

    Returns:
        A summary dict with keys: repo_name, total_files, total_chunks,
        collection_name, duration_seconds.
    """
    start_time = time.time()

    # ── Step 1: Load files ────────────────────────────────────────────
    files = load_files(source, force_reclone=force_reclone)

    # Derive repo name from source
    if source.startswith("http"):
        from src.ingestion.loader import repo_name_from_url
        repo_name = repo_name_from_url(source)
    else:
        repo_name = Path(source).name

    col_name = _collection_name(repo_name)

    # ── Step 2: Check if already indexed ──────────────────────────────
    client = get_chroma_client()
    bm25_path = get_bm25_index_path(repo_name)

    if not force_reindex and bm25_path.exists():
        try:
            existing = client.get_collection(col_name)
            count = existing.count()
            if count > 0:
                print(f"[indexer] Collection '{col_name}' and BM25 index already exist ({count} chunks). "
                      f"Skipping. Use force_reindex=True to re-index.")
                return {
                    "repo_name": repo_name,
                    "total_files": len(files),
                    "total_chunks": count,
                    "collection_name": col_name,
                    "duration_seconds": round(time.time() - start_time, 2),
                    "skipped": True,
                }
        except Exception:
            pass  # Collection doesn't exist yet, proceed

    # Delete existing collection if re-indexing
    try:
        client.delete_collection(col_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )

    # ── Step 3: Chunk all files ───────────────────────────────────────
    print(f"[indexer] Chunking {len(files)} files ...")
    all_chunks: list[Chunk] = []
    for file_info in files:
        chunks = chunk_file(file_info)
        all_chunks.extend(chunks)
    print(f"[indexer] Generated {len(all_chunks)} chunks.")

    if not all_chunks:
        print("[indexer] No chunks to index.")
        return {
            "repo_name": repo_name,
            "total_files": len(files),
            "total_chunks": 0,
            "collection_name": col_name,
            "duration_seconds": round(time.time() - start_time, 2),
            "skipped": False,
        }

    # ── Step 4: Embed and store in batches ────────────────────────────
    model = get_embedding_model()
    batch_size = EMBEDDING_BATCH_SIZE

    print(f"[indexer] Embedding and storing {len(all_chunks)} chunks "
          f"in batches of {batch_size} ...")

    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size

        # Prepare texts for embedding: content + symbol name for richer embeddings
        texts = []
        for chunk in batch:
            # Prepend symbol name and type so the embedding captures
            # "what this code is" not just the raw text
            prefix = ""
            if chunk.symbol_name and chunk.symbol_name != "module":
                prefix = f"{chunk.chunk_type} {chunk.symbol_name}"
                if chunk.parent_class:
                    prefix = f"{chunk.parent_class}.{chunk.symbol_name}"
                prefix += ": "
            texts.append(prefix + chunk.content)

        # Embed the batch
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # Prepare ChromaDB data
        ids = [_chunk_id(chunk, i + j) for j, chunk in enumerate(batch)]
        documents = [chunk.content for chunk in batch]
        metadatas = [
            {
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "symbol_name": chunk.symbol_name,
                "chunk_type": chunk.chunk_type,
                "language": chunk.language,
                "parent_class": chunk.parent_class,
                "decorators": ", ".join(chunk.decorators),
                "has_docstring": chunk.has_docstring,
            }
            for chunk in batch
        ]

        # Upsert into ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        print(f"  Batch {batch_num}/{total_batches} done "
              f"({len(batch)} chunks)")

    # ── Step 5: Build and save BM25 keyword index ─────────────────────
    print(f"[indexer] Building and saving BM25 keyword index ...")
    build_bm25_index(all_chunks, repo_name)

    duration = round(time.time() - start_time, 2)
    print(f"[indexer] ✅ Indexing complete! {len(all_chunks)} chunks "
          f"in {duration}s")

    return {
        "repo_name": repo_name,
        "total_files": len(files),
        "total_chunks": len(all_chunks),
        "collection_name": col_name,
        "duration_seconds": duration,
        "skipped": False,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.indexer <github_url_or_local_path> [--force]")
        sys.exit(1)

    source = sys.argv[1]
    force = "--force" in sys.argv

    result = index_repo(source, force_reindex=force)

    print(f"\n{'='*60}")
    print(f" Repo:       {result['repo_name']}")
    print(f" Files:      {result['total_files']}")
    print(f" Chunks:     {result['total_chunks']}")
    print(f" Collection: {result['collection_name']}")
    print(f" Duration:   {result['duration_seconds']}s")
    if result.get("skipped"):
        print(f" (Skipped — already indexed)")
    print(f"{'='*60}")

