"""LangChain indexer: loads, AST-chunks, enriches, and embeds a repo into Chroma with caching."""

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import DATA_DIR, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from src.ingestion.chunker import chunk_file
from src.ingestion.loader import load_files, repo_name_from_url

CHROMA_DIR = DATA_DIR / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_collection_name(repo_name: str) -> str:
    """Build a Chroma-safe collection name (3-63 chars, alphanumeric/-/_)."""
    clean = "".join(c if c.isalnum() or c in "-_" else "-" for c in repo_name).strip("-_")
    if len(clean) < 3:
        clean += "-idx"
    return clean[:63]


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the cached CPU HuggingFaceEmbeddings singleton."""
    global _embeddings
    if _embeddings is None:
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


def add_metadata_header(doc: Document) -> Document:
    """Prepend '# <type> <parent>.<symbol> | File: <path>' so embeddings, BM25, and reranker see context."""
    m = doc.metadata
    file_path = str(m.get("file_path") or "").replace("\\", "/")
    symbol = str(m.get("symbol_name") or "")
    chunk_type = str(m.get("chunk_type") or "")
    parent = str(m.get("parent_class") or "")

    prefix = ""
    if symbol and symbol != "module":
        prefix = f"{chunk_type} {parent}.{symbol}" if parent else f"{chunk_type} {symbol}"

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
    first, sep, rest = text.partition("\n")
    return rest if (sep and first.startswith("# ") and "File: " in first) else text


def load_documents(repo_name: str) -> list[Document]:
    """Load all indexed Documents directly from Chroma (single source of truth for BM25)."""
    try:
        data = get_vector_store(repo_name).get(include=["documents", "metadatas"])
    except Exception:
        return []
    raw_docs = data.get("documents") or []
    raw_metas = data.get("metadatas") or [{} for _ in raw_docs]
    return [Document(page_content=text, metadata=meta or {}) for text, meta in zip(raw_docs, raw_metas)]


def get_meta_path(repo_name: str) -> Path:
    """Return the path to the repo's fingerprint metadata JSON."""
    return DATA_DIR / f"{repo_name}_meta.json"


def compute_repo_fingerprint(files: list[dict], repo_dir: Optional[Path] = None) -> str:
    """Compute a repo fingerprint from Git HEAD SHA, or MD5 of file paths, sizes, and mtimes."""
    if repo_dir and (repo_dir / ".git").exists():
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(repo_dir), stderr=subprocess.DEVNULL, text=True
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


def _read_cached_meta(repo_name: str) -> dict:
    """Read cached metadata JSON if present, else return an empty dict."""
    meta_path = get_meta_path(repo_name)
    try:
        return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except Exception:
        return {}


def index_repo(source: str, force_reindex: bool = False, force_reclone: bool = False) -> dict:
    """Index a GitHub URL or local folder into Chroma, skipping work when unchanged."""
    start_time = time.time()
    files = load_files(source, force_reclone=force_reclone)
    repo_name = repo_name_from_url(source) if source.startswith("http") else Path(source).name
    col_name = get_collection_name(repo_name)
    repo_dir = Path(source) if Path(source).is_dir() else (DATA_DIR / repo_name)
    fingerprint = compute_repo_fingerprint(files, repo_dir=repo_dir)

    def _summary(total_chunks: int, skipped: bool) -> dict:
        return {
            "repo_name": repo_name, "total_files": len(files), "total_chunks": total_chunks,
            "collection_name": col_name, "duration_seconds": round(time.time() - start_time, 2),
            "skipped": skipped, "fingerprint": fingerprint,
        }

    cached = _read_cached_meta(repo_name)
    if not force_reindex and cached.get("fingerprint") == fingerprint and cached.get("total_chunks", 0) > 0:
        return _summary(cached["total_chunks"], skipped=True)

    all_docs = [add_metadata_header(doc) for f in files for doc in chunk_file(f)]
    if not all_docs:
        return _summary(0, skipped=False)

    vs = get_vector_store(repo_name)
    if force_reindex:
        try:
            vs.delete_collection()
        except Exception:
            pass
        vs = get_vector_store(repo_name)

    # Embed in small batches to stay within 8 GB RAM.
    for i in range(0, len(all_docs), EMBEDDING_BATCH_SIZE):
        vs.add_documents(all_docs[i : i + EMBEDDING_BATCH_SIZE])

    meta = {
        "repo_name": repo_name, "source": str(source), "fingerprint": fingerprint,
        "total_files": len(files), "total_chunks": len(all_docs), "collection_name": col_name,
        "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "embedding_model": EMBEDDING_MODEL,
    }
    get_meta_path(repo_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return _summary(len(all_docs), skipped=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        sys.exit("Usage: python -m src.ingestion.indexer <github_url_or_local_path> [--force]")
    res = index_repo(sys.argv[1], force_reindex="--force" in sys.argv)
    status = "cached" if res["skipped"] else "indexed"
    print(f"{res['repo_name']}: {res['total_files']} files, {res['total_chunks']} chunks ({status} in {res['duration_seconds']}s)")
