"""
Cross-Encoder Reranker module for CodeCompass (Phase 8).

Takes candidate chunks retrieved by Hybrid Search (RRF) and re-scores them using
a lightweight Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~80MB).

Why a Cross-Encoder?
Bi-encoders (like all-MiniLM-L6-v2) embed query and document separately into vectors,
which is fast for retrieval over 100k items but loses subtle cross-attention nuances.
A Cross-Encoder feeds (query, code_chunk) simultaneously into all transformer layers,
allowing full cross-attention between question words and code tokens. This produces
higher ranking accuracy, and running it on just top-20 candidates keeps latency low.

Memory Constraint:
Lazy-loaded on first call to respect the 8 GB RAM limit, with an automatic fallback
to Hybrid RRF if the cross-encoder cannot be loaded.
"""

import argparse
import logging
from typing import Any, Optional

from src.config import RERANKER_MODEL, TOP_K_FINAL, TOP_K_RETRIEVAL
from src.retrieval.hybrid import hybrid_search

logger = logging.getLogger(__name__)

# Global singleton for the cross-encoder model to prevent reloading
_reranker_instance = None


def get_reranker_model(model_name: str = RERANKER_MODEL):
    """
    Lazy-load the CrossEncoder model.

    Tries local cache first (`local_files_only=True`) to avoid network hangs
    on Windows, falling back to download if not cached.
    """
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance

    try:
        from sentence_transformers import CrossEncoder

        try:
            logger.info("Loading cross-encoder reranker from local cache: %s", model_name)
            _reranker_instance = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            logger.info("Cross-encoder not cached locally. Downloading: %s", model_name)
            _reranker_instance = CrossEncoder(model_name)

        return _reranker_instance
    except Exception as e:
        logger.warning(
            "Could not load CrossEncoder model '%s': %s. Falling back to Hybrid RRF without reranking.",
            model_name,
            e,
        )
        return None


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = TOP_K_FINAL,
    model_name: str = RERANKER_MODEL,
) -> list[dict[str, Any]]:
    """
    Rerank candidate chunks using the Cross-Encoder.

    Args:
        query: The user's question.
        candidates: List of chunk dicts from Hybrid search.
        top_k: Number of reranked candidates to return.
        model_name: CrossEncoder model identifier.

    Returns:
        Top-k candidates sorted by cross-encoder relevance score, with updated
        'rank' and 'rerank_score' fields.
    """
    if not candidates or top_k <= 0:
        return []

    model = get_reranker_model(model_name)
    if model is None:
        # Fallback if cross-encoder unavailable
        return [dict(c, rank=i + 1) for i, c in enumerate(candidates[:top_k])]

    # Prepare (query, passage) pairs for cross-attention
    # Include file path and symbol name to give the cross-encoder structural context
    pairs: list[list[str]] = []
    for c in candidates:
        sym = c.get("symbol_name") or ""
        fp = c.get("file_path") or ""
        body = c.get("content") or ""
        passage = f"File: {fp}\nSymbol: {sym}\n{body}"
        pairs.append([query, passage])

    try:
        scores = model.predict(pairs)
    except Exception as e:
        logger.warning("Reranking failed with error: %s. Using RRF order.", e)
        return [dict(c, rank=i + 1) for i, c in enumerate(candidates[:top_k])]

    # Pair candidates with their cross-encoder score
    scored_candidates = []
    for c, score in zip(candidates, scores):
        updated = dict(c)
        updated["rerank_score"] = float(score)
        scored_candidates.append(updated)

    # Sort descending by cross-encoder score
    scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Update rank numbers
    final_results = []
    for idx, item in enumerate(scored_candidates[:top_k], 1):
        item["rank"] = idx
        final_results.append(item)

    return final_results


def hybrid_search_with_rerank(
    query: str,
    repo_name: str,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_final: int = TOP_K_FINAL,
    model_name: str = RERANKER_MODEL,
) -> list[dict[str, Any]]:
    """
    Full two-stage retrieval pipeline:
      Stage 1: Hybrid Search (BM25 + Dense Vector fused via RRF) retrieves top-20.
      Stage 2: Cross-Encoder reranks top-20 down to the top-k chunks.

    Args:
        query: User question.
        repo_name: Indexed repository name.
        top_k_retrieval: Number of candidates retrieved in Stage 1.
        top_k_final: Final number of chunks returned for the prompt.
        model_name: Cross-Encoder model identifier.

    Returns:
        Top-k highest quality chunks.
    """
    # 1. Retrieve candidates via Hybrid RRF
    candidates = hybrid_search(
        query=query,
        repo_name=repo_name,
        top_k_retrieval=top_k_retrieval,
        top_k_final=top_k_retrieval,
    )

    # 2. Rerank top candidates with the cross-encoder
    return rerank(
        query=query,
        candidates=candidates,
        top_k=top_k_final,
        model_name=model_name,
    )


# ── Standalone CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cross-Encoder reranking on a query.")
    parser.add_argument("repo_name", help="Name of the indexed repository")
    parser.add_argument("query", help="User question or query string")
    parser.add_argument("--top-k", type=int, default=TOP_K_FINAL, help="Final top-k chunks")

    args = parser.parse_args()

    print(f"\n======================================================================")
    print(f" 🔀 Hybrid Search + Cross-Encoder Rerank Test")
    print(f" Repo:  {args.repo_name}")
    print(f" Query: {args.query}")
    print(f"======================================================================\n")

    # 1. Hybrid results without rerank
    base_hybrid = hybrid_search(args.query, args.repo_name, top_k_final=args.top_k)
    print(f"--- 1. Hybrid RRF (Base) ---")
    for r in base_hybrid:
        print(f"  #{r['rank']} [RRF score: {r.get('score', 0):.4f}] {r['file_path']}:{r['start_line']}-{r['end_line']} ({r.get('symbol_name')})")

    # 2. Hybrid results with cross-encoder rerank
    reranked = hybrid_search_with_rerank(args.query, args.repo_name, top_k_final=args.top_k)
    print(f"\n--- 2. Hybrid RRF + Cross-Encoder Rerank ---")
    for r in reranked:
        print(f"  #{r['rank']} [Rerank score: {r.get('rerank_score', 0):.4f}] {r['file_path']}:{r['start_line']}-{r['end_line']} ({r.get('symbol_name')})")

