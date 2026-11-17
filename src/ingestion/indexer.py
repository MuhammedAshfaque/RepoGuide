"""
Indexer for CodeCompass using LangChain.

Orchestrates the ingestion pipeline:
  1. Load files from a repo (via loader)
  2. Chunk each file into LangChain Documents (via chunker)
  3. Store embeddings + documents in LangChain Chroma (persistent on disk)
  4. Save documents to disk for LangChain BM25Retriever
  5. Content fingerprint caching to skip re-indexing unchanged repositories

Memory notes (8 GB RAM):
  - HuggingFaceEmbeddings (all-MiniLM-L6-v2) is ~90 MB in memory.
  - Documents are embedded in batches of EMBEDDING_BATCH_SIZE (default 32)
    to keep peak memory low.
  - Chroma runs in-process with persistent storage in data/chroma_db/.
"""

import hashlib
import json
import logging
import pickle
import subprocess
import time
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import (
    DATA_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    get_collection_name,
)
from src.ingestion.loader import load_files
from src.ingestion.chunker import chunk_file

logger = logging.getLogger(__name__)

CHROMA_DIR = DATA_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Global singleton for embeddings
_embeddings_instance: Optional[HuggingFaceEmbeddings] = None


# ── Embeddings & Vector Store ────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get or create the HuggingFaceEmbeddings singleton.

    Uses all-MiniLM-L6-v2 on CPU with local cache priority to avoid
    network checks in offline environments.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        try:
            _embeddings_instance = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu", "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception:
            _embeddings_instance = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return _embeddings_instance


# Alias for backward compatibility
get_embedding_model = get_embeddings


def get_vector_store(repo_name: str) -> Chroma:
    """Get the persistent LangChain Chroma vector store for a repo."""
    col_name = get_collection_name(repo_name)
    return Chroma(
        collection_name=col_name,
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


# ── Document Header Enrichment (for Embeddings, BM25, and Reranker) ──────────

_HEADER_MARKER = " | File: "


def add_metadata_header(doc: Document) -> Document:
    """
    Prepend chunk metadata (chunk_type, parent_class, symbol_name, file_path)
    to page_content so embeddings, BM25, and the cross-encoder see the same
    structural context as manual-v1. Line numbers in metadata stay unchanged.
    """
    m = doc.metadata
    file_path = str(m.get("file_path") or "").replace("\\", "/")
    symbol_name = str(m.get("symbol_name") or "")
    chunk_type = str(m.get("chunk_type") or "")
    parent_class = str(m.get("parent_class") or "")

    prefix = ""
    if symbol_name and symbol_name != "module":
        if parent_class:
            prefix = f"{chunk_type} {parent_class}.{symbol_name}"
        else:
            prefix = f"{chunk_type} {symbol_name}"

    if prefix and file_path:
        header = f"# {prefix}{_HEADER_MARKER}{file_path}"
    elif file_path:
        header = f"# File: {file_path}"
    elif prefix:
        header = f"# {prefix}"
    else:
        return doc

    return Document(
        page_content=f"{header}\n{doc.page_content}",
        metadata=dict(m),
    )


def strip_metadata_header(text: str) -> str:
    """Strip the synthetic metadata header line before displaying raw code in UI/prompts."""
    first_line, sep, rest = text.partition("\n")
    if sep and first_line.startswith("# ") and ("File: " in first_line):
        return rest
    return text


# ── Document Serialization (for BM25Retriever) ───────────────────────────────

def get_docs_path(repo_name: str) -> Path:
    """Return the pickle path for serialized repository documents."""
    return DATA_DIR / f"{repo_name}_docs.pkl"


def save_documents(docs: list[Document], repo_name: str) -> Path:
    """Persist documents to disk so BM25Retriever can be loaded without re-chunking."""
    path = get_docs_path(repo_name)
    with open(path, "wb") as f:
        pickle.dump(docs, f)
    return path


def load_documents(repo_name: str) -> list[Document]:
    """Load serialized documents for a repository."""
    path = get_docs_path(repo_name)
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Metadata & Caching Helpers ───────────────────────────────────────────────

def get_meta_path(repo_name: str) -> Path:
    """Return the path to the repository metadata JSON file."""
    return DATA_DIR / f"{repo_name}_meta.json"


def compute_repo_fingerprint(files: list[dict], repo_dir: Optional[Path] = None) -> str:
    """
    Compute a unique fingerprint for a repository's content.

    If the repository has a .git directory, uses the current commit SHA.
    Otherwise, computes an MD5 checksum over sorted relative file paths, sizes,
    and modification timestamps.
    """
    if repo_dir and (repo_dir / ".git").exists():
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_dir),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            if sha:
                return f"git:{sha}"
        except Exception:
            pass

    hasher = hashlib.md5()
    sorted_files = sorted(files, key=lambda f: f["relative_path"])
    for f in sorted_files:
        p = Path(f["path"])
        mtime = p.stat().st_mtime if p.exists() else 0
        hasher.update(f"{f['relative_path']}:{f['size_kb']}:{mtime}".encode("utf-8"))

    return f"content:{hasher.hexdigest()}"


def is_repo_indexed(repo_name: str) -> bool:
    """Check if both Chroma vector store and serialized documents exist and are non-empty."""
    docs_path = get_docs_path(repo_name)
    if not docs_path.exists():
        # Fallback check for old manual format
        old_bm25 = DATA_DIR / f"{repo_name}_bm25.pkl"
        if not old_bm25.exists():
            return False

    try:
        vs = get_vector_store(repo_name)
        return vs._collection.count() > 0
    except Exception:
        return False


# ── Main Indexing Pipeline ───────────────────────────────────────────────────

def index_repo(
    source: str,
    force_reindex: bool = False,
    force_reclone: bool = False,
) -> dict:
    """
    Full indexing pipeline: load -> chunk -> embed in Chroma -> cache documents.

    Args:
        source: GitHub URL or local directory path.
        force_reindex: If True, delete existing index and re-index from scratch.
        force_reclone: If True, re-clone repository even if it exists locally.

    Returns:
        Summary dictionary with repo_name, total_files, total_chunks,
        collection_name, duration_seconds, skipped, fingerprint.
    """
    start_time = time.time()

    # Step 1: Load files
    files = load_files(source, force_reclone=force_reclone)

    if source.startswith("http"):
        from src.ingestion.loader import repo_name_from_url
        repo_name = repo_name_from_url(source)
    else:
        repo_name = Path(source).name

    col_name = get_collection_name(repo_name)
    repo_dir = Path(source) if Path(source).is_dir() else (DATA_DIR / repo_name)
    current_fingerprint = compute_repo_fingerprint(files, repo_dir=repo_dir)
    meta_path = get_meta_path(repo_name)

    # Step 2: Check fingerprint cache
    if not force_reindex and is_repo_indexed(repo_name):
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("fingerprint") == current_fingerprint:
                    print(f"[indexer] Repo '{repo_name}' unchanged (fingerprint match). Skipping re-index.")
                    vs = get_vector_store(repo_name)
                    return {
                        "repo_name": repo_name,
                        "total_files": len(files),
                        "total_chunks": vs._collection.count(),
                        "collection_name": col_name,
                        "duration_seconds": round(time.time() - start_time, 2),
                        "skipped": True,
                        "fingerprint": current_fingerprint,
                    }
            except Exception:
                pass

    # Step 3: Chunk all files into LangChain Documents (with metadata header for retrieval)
    print(f"[indexer] Chunking {len(files)} files into LangChain Documents ...")
    all_docs: list[Document] = []
    for file_info in files:
        chunks = chunk_file(file_info)
        all_docs.extend(add_metadata_header(c) for c in chunks)
    print(f"[indexer] Generated {len(all_docs)} LangChain Documents.")

    if not all_docs:
        return {
            "repo_name": repo_name,
            "total_files": len(files),
            "total_chunks": 0,
            "collection_name": col_name,
            "duration_seconds": round(time.time() - start_time, 2),
            "skipped": False,
            "fingerprint": current_fingerprint,
        }

    # Step 4: Index into LangChain Chroma
    vs = get_vector_store(repo_name)
    if force_reindex:
        try:
            vs.delete_collection()
        except Exception:
            pass
        vs = get_vector_store(repo_name)

    batch_size = EMBEDDING_BATCH_SIZE
    total_batches = (len(all_docs) + batch_size - 1) // batch_size
    print(f"[indexer] Embedding {len(all_docs)} chunks into Chroma in batches of {batch_size} ...")

    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        vs.add_documents(batch)
        print(f"  Batch {batch_num}/{total_batches} done ({len(batch)} chunks)")

    # Step 5: Save documents for BM25Retriever
    save_documents(all_docs, repo_name)

    # Step 6: Save metadata with fingerprint
    meta_data = {
        "repo_name": repo_name,
        "source": str(source),
        "fingerprint": current_fingerprint,
        "total_files": len(files),
        "total_chunks": len(all_docs),
        "collection_name": col_name,
        "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": EMBEDDING_MODEL,
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2)
    except Exception as e:
        logger.warning("Could not write metadata cache file: %s", e)

    duration = round(time.time() - start_time, 2)
    print(f"[indexer] ✅ Indexing complete! {len(all_docs)} chunks in {duration}s")

    return {
        "repo_name": repo_name,
        "total_files": len(files),
        "total_chunks": len(all_docs),
        "collection_name": col_name,
        "duration_seconds": duration,
        "skipped": False,
        "fingerprint": current_fingerprint,
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
        print(f" (Skipped — unchanged cache match)")
    print(f"{'='*60}")
