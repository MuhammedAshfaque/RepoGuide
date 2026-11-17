"""LangChain retrieval pipelines: Vector (Chroma), BM25, Hybrid (RRF), and optional Cross-Encoder Rerank."""

import logging
import re
from typing import Optional

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda

from src.config import RERANKER_MODEL, TOP_K_FINAL, TOP_K_RETRIEVAL, get_collection_name
from src.ingestion.indexer import get_vector_store, load_documents

logger = logging.getLogger(__name__)

_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_bm25_cache: dict[tuple[str, int], BM25Retriever] = {}
_cross_encoder_cache: dict[str, HuggingFaceCrossEncoder] = {}


def clear_retriever_cache() -> None:
    """Clear cached BM25 retrievers (call after re-indexing or in tests)."""
    _bm25_cache.clear()


def split_identifier(identifier: str) -> list[str]:
    """Split snake_case, camelCase, and PascalCase identifiers into lowercased subwords."""
    parts = re.split(r"[_\s]+", _CAMEL_RE.sub(r"\1 \2", identifier))
    return [p.lower() for p in parts if p.strip()]


def tokenize_code(text: str) -> list[str]:
    """Tokenize code/queries while keeping both full identifiers and split subwords."""
    tokens: list[str] = []
    for tok in re.findall(r"[A-Za-z0-9_]+|[^\s\w]", text):
        tok = tok.strip()
        if not tok:
            continue
        lowered = tok.lower()
        tokens.append(lowered)
        if "_" in tok or _CAMEL_RE.search(tok):
            tokens.extend(sub for sub in split_identifier(tok) if sub and sub != lowered)
    return tokens


def get_vector_retriever(repo_name: str, top_k: int = TOP_K_FINAL):
    """Return a LangChain Chroma vector retriever."""
    return get_vector_store(repo_name).as_retriever(search_kwargs={"k": top_k})


def get_bm25_retriever(repo_name: str, top_k: int = TOP_K_FINAL) -> BM25Retriever:
    """Return a cached LangChain BM25Retriever using identifier-aware tokenization."""
    cache_key = (get_collection_name(repo_name), top_k)
    if cache_key in _bm25_cache:
        return _bm25_cache[cache_key]

    docs = load_documents(repo_name)
    if not docs:
        raise FileNotFoundError(f"No indexed documents found for '{repo_name}'. Run index_repo() first.")

    retriever = BM25Retriever.from_documents(docs, preprocess_func=tokenize_code, k=top_k)
    _bm25_cache[cache_key] = retriever
    return retriever


def get_hybrid_retriever(
    repo_name: str,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_final: int = TOP_K_FINAL,
) -> Runnable[str, list[Document]]:
    """Return an LCEL hybrid RRF pipeline (equal 0.5/0.5 weights, c=60, sliced to top_k_final)."""
    ensemble = EnsembleRetriever(
        retrievers=[
            get_vector_retriever(repo_name, top_k=top_k_retrieval),
            get_bm25_retriever(repo_name, top_k=top_k_retrieval),
        ],
        weights=[0.5, 0.5],
        c=60,
    )
    return ensemble | RunnableLambda(lambda docs: docs[:top_k_final])


def get_cross_encoder(model_name: str = RERANKER_MODEL) -> Optional[HuggingFaceCrossEncoder]:
    """Lazy-load and cache the cross-encoder (~80 MB CPU); returns None if unavailable."""
    if model_name in _cross_encoder_cache:
        return _cross_encoder_cache[model_name]

    try:
        try:
            encoder = HuggingFaceCrossEncoder(model_name=model_name, model_kwargs={"local_files_only": True})
        except Exception:
            encoder = HuggingFaceCrossEncoder(model_name=model_name)
        _cross_encoder_cache[model_name] = encoder
        return encoder
    except Exception as exc:
        logger.warning("CrossEncoder '%s' unavailable (%s); falling back to hybrid.", model_name, exc)
        return None


def get_rerank_retriever(
    repo_name: str,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_final: int = TOP_K_FINAL,
    model_name: str = RERANKER_MODEL,
) -> Runnable[str, list[Document]]:
    """Return a two-stage Hybrid (top-20) -> CrossEncoderReranker (top-k) retriever, falling back to Hybrid."""
    cross_encoder = get_cross_encoder(model_name)
    if cross_encoder is None:
        return get_hybrid_retriever(repo_name, top_k_retrieval=top_k_retrieval, top_k_final=top_k_final)

    base_hybrid = get_hybrid_retriever(repo_name, top_k_retrieval=top_k_retrieval, top_k_final=top_k_retrieval)
    return ContextualCompressionRetriever(
        base_compressor=CrossEncoderReranker(model=cross_encoder, top_n=top_k_final),
        base_retriever=base_hybrid,
    )


def get_retriever(
    repo_name: str,
    mode: str = "hybrid",
    top_k: int = TOP_K_FINAL,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
) -> Runnable[str, list[Document]]:
    """Return a LangChain retriever Runnable for 'vector', 'bm25', 'hybrid', or 'hybrid_rerank'."""
    mode = mode.lower().strip()
    if mode == "vector":
        return get_vector_retriever(repo_name, top_k=top_k)
    if mode == "bm25":
        return get_bm25_retriever(repo_name, top_k=top_k)
    if mode == "hybrid":
        return get_hybrid_retriever(repo_name, top_k_retrieval=top_k_retrieval, top_k_final=top_k)
    if mode == "hybrid_rerank":
        return get_rerank_retriever(repo_name, top_k_retrieval=top_k_retrieval, top_k_final=top_k)
    raise ValueError(f"Unknown retrieval mode '{mode}'. Choose from: vector, bm25, hybrid, hybrid_rerank.")
