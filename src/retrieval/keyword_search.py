"""
Keyword search using BM25 for CodeCompass.

Implements BM25Okapi retrieval with identifier-aware tokenization.
Code contains compound words in snake_case (e.g. `get_user_by_id`) and
camelCase (e.g. `UserProfileView`). Standard tokenizers treat these as single
opaque tokens; our code-aware tokenizer preserves both the compound token
and its individual constituent words.

Why BM25 alongside Vector Search?
  Vector search is great at understanding broad semantic intent
  ("Where is authentication handled?"), but struggles with exact symbol names,
  rare keywords, and error strings ("jwt_optional", "404_NOT_FOUND").
  BM25 provides exact term-frequency matching that complements dense retrieval.
"""

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.config import DATA_DIR, TOP_K_RETRIEVAL, get_collection_name
from src.ingestion.chunker import Chunk


# ── Code-Aware Tokenization ──────────────────────────────────────────────────

_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+|[^\s\w]")


def split_identifier(identifier: str) -> list[str]:
    """
    Split a programming identifier into subwords.

    Handles:
      - snake_case: 'get_user_by_id' → ['get', 'user', 'by', 'id']
      - camelCase:  'getUserById'    → ['get', 'user', 'by', 'id']
      - PascalCase: 'UserProfile'    → ['user', 'profile']
      - SCREAMING:  'MAX_SIZE_KB'    → ['max', 'size', 'kb']

    Returns all subwords lowercased.
    """
    # Split camelCase / PascalCase transitions: 'getUser' -> 'get User'
    s = _CAMEL_RE.sub(r"\1 \2", identifier)
    # Split on underscores and whitespace
    parts = re.split(r"[_\s]+", s)
    return [p.lower() for p in parts if p.strip()]


def tokenize_code(text: str) -> list[str]:
    """
    Tokenize code or query text with identifier awareness.

    Produces both the original compound token and its constituent subwords so
    queries can match either the full identifier or any of its components.

    Example:
      'def get_user_by_id(user_id):'
      → ['def', 'get_user_by_id', 'get', 'user', 'by', 'id',
         'user_id', 'user', 'id']
    """
    tokens: list[str] = []
    # Match alphanumeric/underscore chunks or punctuation symbols
    raw_tokens = re.findall(r"[A-Za-z0-9_]+|[^\s\w]", text)

    for tok in raw_tokens:
        tok_clean = tok.strip()
        if not tok_clean:
            continue

        lowered = tok_clean.lower()
        tokens.append(lowered)

        # If it looks like an identifier with multiple parts, add subwords
        if "_" in tok_clean or _CAMEL_RE.search(tok_clean):
            subwords = split_identifier(tok_clean)
            for sub in subwords:
                if sub and sub != lowered:
                    tokens.append(sub)

    return tokens


# ── BM25 Index Container & Persistence ───────────────────────────────────────

class BM25Index:
    """Stores the BM25 model alongside chunk metadata dicts."""

    def __init__(self, bm25: BM25Okapi, chunk_records: list[dict[str, Any]]):
        self.bm25 = bm25
        self.chunk_records = chunk_records

    def save(self, path: Path) -> None:
        """Persist index to disk using pickle."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Load index from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if obj.__class__.__name__ != "BM25Index":
            raise TypeError(f"Expected BM25Index, got {type(obj)}")
        return obj


def get_bm25_index_path(repo_name: str) -> Path:
    """Return the file path for a repo's serialized BM25 index."""
    col_name = get_collection_name(repo_name)
    return DATA_DIR / f"{col_name}_bm25.pkl"


# Cache for in-memory BM25 index across calls
_bm25_cache: dict[str, BM25Index] = {}


def build_bm25_index(chunks: list[Chunk], repo_name: str) -> Path:
    """
    Build and persist a BM25 index over a list of chunks.

    Args:
        chunks: List of Chunk objects from the chunker.
        repo_name: Name of the repository.

    Returns:
        Path to the saved .pkl index file.
    """
    corpus_tokens: list[list[str]] = []
    chunk_records: list[dict[str, Any]] = []

    for chunk in chunks:
        # Prepend symbol name and file path to give extra retrieval signal
        symbol_text = f"{chunk.symbol_name} {chunk.parent_class}" if chunk.symbol_name else ""
        doc_text = f"{symbol_text}\n{chunk.file_path}\n{chunk.content}"

        tokens = tokenize_code(doc_text)
        corpus_tokens.append(tokens)

        chunk_records.append({
            "content": chunk.content,
            "file_path": chunk.file_path,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "symbol_name": chunk.symbol_name,
            "chunk_type": chunk.chunk_type,
            "language": chunk.language,
            "parent_class": chunk.parent_class,
            "decorators": ", ".join(chunk.decorators),
            "has_docstring": chunk.has_docstring,
        })

    bm25 = BM25Okapi(corpus_tokens)
    index_obj = BM25Index(bm25=bm25, chunk_records=chunk_records)

    index_path = get_bm25_index_path(repo_name)
    index_obj.save(index_path)

    # Update in-memory cache
    col_name = get_collection_name(repo_name)
    _bm25_cache[col_name] = index_obj

    print(f"[keyword_search] BM25 index built with {len(chunks)} chunks -> {index_path.name}")
    return index_path


def load_bm25_index(repo_name: str) -> BM25Index:
    """Load the BM25 index for a repo, utilizing in-memory cache if available."""
    col_name = get_collection_name(repo_name)
    if col_name in _bm25_cache:
        return _bm25_cache[col_name]

    path = get_bm25_index_path(repo_name)
    if not path.exists():
        raise FileNotFoundError(
            f"No BM25 index found at {path}. "
            f"Please run index_repo() first."
        )

    index_obj = BM25Index.load(path)
    _bm25_cache[col_name] = index_obj
    return index_obj


# ── BM25 Search Function ─────────────────────────────────────────────────────

def bm25_search(
    query: str,
    repo_name: str,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict[str, Any]]:
    """
    Search code chunks using BM25 keyword matching.

    Args:
        query: User search query or code identifier.
        repo_name: Name of the repository.
        top_k: Number of results to return.

    Returns:
        List of result dicts sorted by BM25 score descending, matching
        the structure of vector_search results.
    """
    index_obj = load_bm25_index(repo_name)
    query_tokens = tokenize_code(query)

    if not query_tokens:
        return []

    scores = index_obj.bm25.get_scores(query_tokens)

    # Pair scores with chunk records
    scored_items = list(zip(scores, index_obj.chunk_records))
    # Filter out items with 0 score (no keyword match at all)
    positive_items = [(s, r) for s, r in scored_items if s > 0.0]

    # Sort descending by score
    positive_items.sort(key=lambda x: x[0], reverse=True)
    top_items = positive_items[:top_k]

    results: list[dict[str, Any]] = []
    for rank, (score, record) in enumerate(top_items, 1):
        res = dict(record)
        res["score"] = round(float(score), 4)
        res["rank"] = rank
        results.append(res)

    return results


# ── CLI for testing ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.retrieval.keyword_search <repo_name> <query>")
        print('Example: python -m src.retrieval.keyword_search flask-realworld-example-app "jwt_optional"')
        sys.exit(1)

    repo_name = sys.argv[1]
    query = " ".join(sys.argv[2:])

    print(f"\n🔎 BM25 Query: \"{query}\"")
    print(f"📦 Repo:       {repo_name}")
    print(f"{'='*70}")

    results = bm25_search(query, repo_name, top_k=10)

    if not results:
        print("  No keyword matches found.")
    else:
        for r in results:
            symbol = r["symbol_name"] or "(block)"
            parent = f" in {r['parent_class']}" if r["parent_class"] else ""
            citation = f"{r['file_path']}:{r['start_line']}-{r['end_line']}"
            print(f"\n  #{r['rank']}  BM25 score={r['score']:.4f}  {r['chunk_type']} '{symbol}'{parent}")
            print(f"       📄 {citation}")
            preview = r["content"].splitlines()[:3]
            for line in preview:
                print(f"       {line.rstrip()}")
            if len(r["content"].splitlines()) > 3:
                print(f"       ... ({len(r['content'].splitlines()) - 3} more lines)")
