"""Repository indexer: chunks files, enriches headers, embeds into Chroma, and caches by fingerprint."""

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import DATA_DIR, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL, get_collection_name
from src.ingestion.chunker import chunk_file
from src.ingestion.loader import load_files, repo_name_from_url

logger = logging.getLogger(__name__)

CHROMA_DIR = DATA_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

_embeddings: Optional[HuggingFaceEmbeddings] = None


# ── Embeddings & Vector Store ────────────────────────────────────────────────

def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazy-load the CPU HuggingFaceEmbeddings singleton (offline cache first)."""
    global _embeddings
    if _embeddings is None:
        try:
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu", "local_files_only": True},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception:
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    return _embeddings


def get_vector_store(repo_name: str) -> Chroma:
    """Return the persistent LangChain Chroma store for a repository."""
    return Chroma(
        collection_name=get_collection_name(repo_name),
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


def load_documents(repo_name: str) -> list[Document]:
    """Load indexed Documents directly from Chroma (used by BM25Retriever)."""
    try:
        data = get_vector_store(repo_name).get(include=["documents", "metadatas"])
    except Exception:
        return []
    texts = data.get("documents") or []
    metas = data.get("metadatas") or []
    return [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metas)]


# ── Metadata Header Enrichment ───────────────────────────────────────────────

def add_metadata_header(doc: Document) -> Document:
    """Prepend '# <type> <parent>.<symbol> | File: <path>' so all retrievers see file/symbol context."""
    m = doc.metadata
    file_path = str(m.get("file_path") or "").replace("\\", "/")
    symbol = str(m.get("symbol_name") or "")
    ctype = str(m.get("chunk_type") or "")
    parent = str(m.get("parent_class") or "")

    prefix = ""
    if symbol and symbol != "module":
        prefix = f"{ctype} {parent}.{symbol}" if parent else f"{ctype} {symbol}"

    if prefix and file_path:
        header = f"# {prefix} | File: {file_path}"
    elif file_path:
        header = f"# File: {file_path}"
    elif prefix:
        header = f"# {prefix}"
    else:
        return doc

    return Document(page_content=f"{header}\n{doc.page_content}", metadata=dict(m))


def strip_metadata_header(text: str) -> str:
    """Remove the synthetic metadata header line before showing raw code in prompts or UI."""
    first_line, sep, rest = text.partition("\n")
    if sep and first_line.startswith("# ") and "File: " in first_line:
        return rest
    return text


# ── Fingerprint Caching Helpers ──────────────────────────────────────────────

def get_meta_path(repo_name: str) -> Path:
    """Return the path to the repository index metadata JSON file."""
    return DATA_DIR / f"{repo_name}_meta.json"


def compute_repo_fingerprint(files: list[dict], repo_dir: Optional[Path] = None) -> str:
    """Compute a Git HEAD SHA (if available) or MD5 checksum of file paths, sizes, and mtimes."""
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
    for f in sorted(files, key=lambda x: x["relative_path"]):
        p = Path(f["path"])
        mtime = p.stat().st_mtime if p.exists() else 0
        hasher.update(f"{f['relative_path']}:{f['size_kb']}:{mtime}".encode("utf-8"))
    return f"content:{hasher.hexdigest()}"


def is_repo_indexed(repo_name: str) -> bool:
    """Return True if metadata exists and Chroma has at least one stored chunk."""
    if not get_meta_path(repo_name).exists():
        return False
    try:
        return get_vector_store(repo_name)._collection.count() > 0
    except Exception:
        return False


def _cache_matches(repo_name: str, fingerprint: str) -> bool:
    """Return True if the saved index metadata matches the current repo fingerprint."""
    meta_path = get_meta_path(repo_name)
    if not meta_path.exists() or not is_repo_indexed(repo_name):
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta.get("fingerprint") == fingerprint
    except Exception:
        return False


# ── Main Indexing Pipeline ───────────────────────────────────────────────────

def index_repo(source: str, force_reindex: bool = False, force_reclone: bool = False) -> dict:
    """Load, chunk, enrich, and embed a repository into Chroma with fingerprint caching."""
    start_time = time.time()
    files = load_files(source, force_reclone=force_reclone)
    repo_name = repo_name_from_url(source) if source.startswith("http") else Path(source).name
    col_name = get_collection_name(repo_name)
    repo_dir = Path(source) if Path(source).is_dir() else (DATA_DIR / repo_name)
    fingerprint = compute_repo_fingerprint(files, repo_dir=repo_dir)

    if not force_reindex and _cache_matches(repo_name, fingerprint):
        print(f"[indexer] Repo '{repo_name}' unchanged (fingerprint match). Skipping re-index.")
        return {
            "repo_name": repo_name,
            "total_files": len(files),
            "total_chunks": get_vector_store(repo_name)._collection.count(),
            "collection_name": col_name,
            "duration_seconds": round(time.time() - start_time, 2),
            "skipped": True,
            "fingerprint": fingerprint,
        }

    docs = [add_metadata_header(c) for f in files for c in chunk_file(f)]
    if not docs:
        return {
            "repo_name": repo_name, "total_files": len(files), "total_chunks": 0,
            "collection_name": col_name, "duration_seconds": round(time.time() - start_time, 2),
            "skipped": False, "fingerprint": fingerprint,
        }

    vs = get_vector_store(repo_name)
    if force_reindex:
        try:
            vs.delete_collection()
        except Exception:
            pass
        vs = get_vector_store(repo_name)

    # Batch embeds to stay comfortably inside 8 GB RAM
    for i in range(0, len(docs), EMBEDDING_BATCH_SIZE):
        vs.add_documents(docs[i : i + EMBEDDING_BATCH_SIZE])

    meta = {
        "repo_name": repo_name,
        "source": str(source),
        "fingerprint": fingerprint,
        "total_files": len(files),
        "total_chunks": len(docs),
        "collection_name": col_name,
        "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": EMBEDDING_MODEL,
    }
    try:
        get_meta_path(repo_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not write metadata cache file: %s", exc)

    duration = round(time.time() - start_time, 2)
    print(f"[indexer] Indexed {len(docs)} chunks from {len(files)} files in {duration}s")
    return {
        "repo_name": repo_name,
        "total_files": len(files),
        "total_chunks": len(docs),
        "collection_name": col_name,
        "duration_seconds": duration,
        "skipped": False,
        "fingerprint": fingerprint,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.indexer <github_url_or_local_path> [--force]")
        sys.exit(1)
    result = index_repo(sys.argv[1], force_reindex="--force" in sys.argv)
    print(result)
