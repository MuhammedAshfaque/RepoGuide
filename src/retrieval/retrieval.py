"""
Consolidated retrieval module for CodeCompass using LangChain.

Supports four retrieval modes over an indexed repository:
  1. vector        — Semantic search via LangChain Chroma (all-MiniLM-L6-v2)
  2. bm25          — Keyword search via BM25Retriever with identifier-aware tokenization
  3. hybrid        — Reciprocal Rank Fusion via EnsembleRetriever (equal 0.5/0.5 weights)
  4. hybrid_rerank — Two-stage Hybrid RRF (top-20) + CrossEncoderReranker (optional)

Memory notes (8 GB RAM):
  - The cross-encoder (~80 MB) is lazy-loaded only when 'hybrid_rerank' is requested.
  - If the cross-encoder cannot be loaded, 'hybrid_rerank' falls back to 'hybrid'.
  - All modes return at most top_k (default 4) Documents to keep LLM context small.
"""

import logging
import re
from typing import Optional

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from src.config import (
    RERANKER_MODEL,
    TOP_K_FINAL,
    TOP_K_RETRIEVAL,
    get_collection_name,
)
from src.ingestion.indexer import get_vector_store, load_documents

logger = logging.getLogger(__name__)

RRF_C_DEFAULT = 60

# Module-level caches to avoid reloading models/indexes on every query
_bm25_cache: dict[tuple[str, int], BM25Retriever] = {}
_cross_encoder_cache: dict[str, HuggingFaceCrossEncoder] = {}


def clear_retriever_cache() -> None:
    """Clear cached BM25 retrievers (useful after re-indexing or in tests)."""
    _bm25_cache.clear()


# ── Identifier-Aware Tokenization (for BM25Retriever) ────────────────────────

_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")


def split_identifier(identifier: str) -> list[str]:
    """
    Split a programming identifier into lowercased subwords.

    Examples:
      - 'get_user_by_id' -> ['get', 'user', 'by', 'id']
      - 'getUserById'    -> ['get', 'user', 'by', 'id']
      - 'UserProfile'    -> ['user', 'profile']
      - 'MAX_SIZE_KB'    -> ['max', 'size', 'kb']
    """
    s = _CAMEL_RE.sub(r"\1 \2", identifier)
    parts = re.split(r"[_\s]+", s)
    return [p.lower() for p in parts if p.strip()]


def tokenize_code(text: str) -> list[str]:
    """
    Tokenize code or query text while preserving compound identifiers and subwords.

    Passed as `preprocess_func` to LangChain's BM25Retriever so queries match
    both exact symbols (e.g. 'jwt_optional') and subwords (e.g. 'jwt').
    """
    tokens: list[str] = []
    raw_tokens = re.findall(r"[A-Za-z0-9_]+|[^\s\w]", text)

    for tok in raw_tokens:
        tok_clean = tok.strip()
        if not tok_clean:
            continue

        lowered = tok_clean.lower()
        tokens.append(lowered)

        if "_" in tok_clean or _CAMEL_RE.search(tok_clean):
            for sub in split_identifier(tok_clean):
                if sub and sub != lowered:
                    tokens.append(sub)

    return tokens


# ── Retriever Builders ───────────────────────────────────────────────────────

def get_vector_retriever(repo_name: str, top_k: int = TOP_K_FINAL):
    """Return a LangChain Chroma vector retriever for the repository."""
    vs = get_vector_store(repo_name)
    return vs.as_retriever(search_kwargs={"k": top_k})


def get_bm25_retriever(repo_name: str, top_k: int = TOP_K_FINAL) -> BM25Retriever:
    """
    Return a LangChain BM25Retriever using our identifier-aware tokenizer.

    Loads serialized Documents saved by the indexer and caches the retriever.
    """
    col_name = get_collection_name(repo_name)
    cache_key = (col_name, top_k)
    if cache_key in _bm25_cache:
        return _bm25_cache[cache_key]

    docs = load_documents(repo_name)
    if not docs:
        raise FileNotFoundError(
            f"No indexed documents found for '{repo_name}'. "
            "Please run index_repo() first."
        )

    retriever = BM25Retriever.from_documents(
        docs,
        preprocess_func=tokenize_code,
        k=top_k,
    )
    _bm25_cache[cache_key] = retriever
    return retriever


def get_hybrid_retriever(
    repo_name: str,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_final: int = TOP_K_FINAL,
) -> Runnable[str, list[Document]]:
    """
    Return a Hybrid RRF retriever combining Vector + BM25 via EnsembleRetriever.

    Fetches `top_k_retrieval` (default 20) candidates from each retriever,
    fuses them with equal weights [0.5, 0.5] and RRF constant c=60,
    and slices to `top_k_final` results.
    """
    vec_retriever = get_vector_retriever(repo_name, top_k=top_k_retrieval)
    bm25_retriever = get_bm25_retriever(repo_name, top_k=top_k_retrieval)

    ensemble = EnsembleRetriever(
        retrievers=[vec_retriever, bm25_retriever],
        weights=[0.5, 0.5],
        c=RRF_C_DEFAULT,
    )
    # EnsembleRetriever returns all fused unique docs; slice to top_k_final
    return ensemble | RunnableLambda(lambda docs: docs[:top_k_final])


def get_cross_encoder(model_name: str = RERANKER_MODEL) -> Optional[HuggingFaceCrossEncoder]:
    """
    Lazy-load and cache the HuggingFaceCrossEncoder (~80 MB on CPU).

    Tries local HuggingFace cache first to avoid offline hangs.
    Returns None if loading fails so callers can gracefully fall back to hybrid.
    """
    if model_name in _cross_encoder_cache:
        return _cross_encoder_cache[model_name]

    try:
        try:
            encoder = HuggingFaceCrossEncoder(
                model_name=model_name,
                model_kwargs={"local_files_only": True},
            )
        except Exception:
            encoder = HuggingFaceCrossEncoder(model_name=model_name)

        _cross_encoder_cache[model_name] = encoder
        return encoder
    except Exception as e:
        logger.warning(
            "Could not load CrossEncoder '%s': %s. Falling back to hybrid search.",
            model_name,
            e,
        )
        return None


def get_rerank_retriever(
    repo_name: str,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_final: int = TOP_K_FINAL,
    model_name: str = RERANKER_MODEL,
) -> Runnable[str, list[Document]]:
    """
    Return a two-stage Hybrid + Cross-Encoder Rerank retriever.

    Stage 1: Hybrid RRF fetches top `top_k_retrieval` (default 20) candidates.
    Stage 2: CrossEncoderReranker re-scores and keeps the top `top_k_final`.
    Falls back to plain Hybrid RRF if the cross-encoder cannot be loaded.
    """
    cross_encoder = get_cross_encoder(model_name)
    if cross_encoder is None:
        return get_hybrid_retriever(
            repo_name,
            top_k_retrieval=top_k_retrieval,
            top_k_final=top_k_final,
        )

    base_hybrid = get_hybrid_retriever(
        repo_name,
        top_k_retrieval=top_k_retrieval,
        top_k_final=top_k_retrieval,
    )
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=top_k_final)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_hybrid,
    )


def get_retriever(
    repo_name: str,
    mode: str = "hybrid",
    top_k: int = TOP_K_FINAL,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
) -> Runnable[str, list[Document]]:
    """
    Factory returning a LangChain retriever Runnable for the requested mode.

    Args:
        repo_name: Indexed repository name.
        mode: One of 'vector', 'bm25', 'hybrid', 'hybrid_rerank'.
        top_k: Final number of Documents to return (default 4).
        top_k_retrieval: Candidates fetched per sub-retriever in hybrid modes (default 20).
    """
    mode = mode.lower().strip()
    if mode == "vector":
        return get_vector_retriever(repo_name, top_k=top_k)
    if mode == "bm25":
        return get_bm25_retriever(repo_name, top_k=top_k)
    if mode == "hybrid":
        return get_hybrid_retriever(
            repo_name,
            top_k_retrieval=top_k_retrieval,
            top_k_final=top_k,
        )
    if mode == "hybrid_rerank":
        return get_rerank_retriever(
            repo_name,
            top_k_retrieval=top_k_retrieval,
            top_k_final=top_k,
        )
    raise ValueError(
        f"Unknown retrieval mode '{mode}'. "
        "Choose from: 'vector', 'bm25', 'hybrid', 'hybrid_rerank'."
    )


# ── CLI for quick testing ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test LangChain retrieval modes.")
    parser.add_argument("repo_name", help="Name of the indexed repository")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--mode",
        choices=["vector", "bm25", "hybrid", "hybrid_rerank"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K_FINAL, help="Results to return")
    args = parser.parse_args()

    retriever = get_retriever(args.repo_name, mode=args.mode, top_k=args.top_k)
    docs = retriever.invoke(args.query)

    print(f"\n🔎 Mode: {args.mode} | Repo: {args.repo_name} | Query: \"{args.query}\"")
    print("=" * 70)
    for rank, doc in enumerate(docs, 1):
        m = doc.metadata
        sym = m.get("symbol_name") or "(block)"
        loc = f"{m.get('file_path')}:{m.get('start_line')}-{m.get('end_line')}"
        print(f"  #{rank}  {m.get('chunk_type', 'code')} '{sym}'  |  {loc}")
    print("=" * 70)
