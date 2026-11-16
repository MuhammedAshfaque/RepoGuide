"""
Hybrid search with Reciprocal Rank Fusion (RRF) for CodeCompass.

Combines dense vector search (semantic similarity) and sparse BM25 search
(exact keyword and identifier matching).

How Reciprocal Rank Fusion (RRF) works:
  Given ranked lists from multiple retrievers, the score for each document d is:
      RRF_score(d) = SUM_{m in models} 1 / (k + rank_m(d))
  where k is a smoothing constant (standard default: 60), and rank_m(d) is the
  1-based rank position of document d in retriever m's results.

Why RRF over score combination?
  Vector scores (cosine similarity in [0, 1]) and BM25 scores (unbounded TF-IDF)
  are on completely different scales and distributions. Normalizing and weighting
  them requires sensitive tuning that changes across queries.
  RRF is purely rank-based, parameter-free, scale-invariant, and proven to
  consistently outperform either retriever alone.
"""

from typing import Any

from src.config import TOP_K_RETRIEVAL, TOP_K_FINAL
from src.retrieval.vector_search import vector_search
from src.retrieval.keyword_search import bm25_search

RRF_K_DEFAULT = 60


def _chunk_key(chunk_dict: dict[str, Any]) -> tuple[str, int, int]:
    """Generate a unique identifying tuple for a chunk."""
    return (
        chunk_dict["file_path"],
        int(chunk_dict["start_line"]),
        int(chunk_dict["end_line"]),
    )


def compute_rrf_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    rrf_k: int = RRF_K_DEFAULT,
) -> list[dict[str, Any]]:
    """
    Apply Reciprocal Rank Fusion across multiple ranked lists of results.

    Args:
        ranked_lists: A list of ranked result lists, each sorted by score descending.
        rrf_k: Smoothing constant (default: 60).

    Returns:
        Combined list of unique chunks, sorted by RRF score descending.
        Each item includes 'rrf_score' and metadata on its individual ranks.
    """
    scores: dict[tuple[str, int, int], float] = {}
    chunk_store: dict[tuple[str, int, int], dict[str, Any]] = {}
    source_ranks: dict[tuple[str, int, int], dict[int, int]] = {}

    for list_idx, result_list in enumerate(ranked_lists):
        for rank_zero_based, item in enumerate(result_list):
            rank_one_based = rank_zero_based + 1
            key = _chunk_key(item)

            if key not in chunk_store:
                chunk_store[key] = dict(item)
                source_ranks[key] = {}

            # Add RRF contribution: 1 / (k + rank)
            contribution = 1.0 / (rrf_k + rank_one_based)
            scores[key] = scores.get(key, 0.0) + contribution
            source_ranks[key][list_idx] = rank_one_based

    # Sort keys by accumulated RRF score descending
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    fused_results: list[dict[str, Any]] = []
    for rank, key in enumerate(sorted_keys, 1):
        item = dict(chunk_store[key])
        item["score"] = round(scores[key], 6)
        item["rrf_score"] = round(scores[key], 6)
        item["rank"] = rank
        item["vector_rank"] = source_ranks[key].get(0)
        item["bm25_rank"] = source_ranks[key].get(1)
        fused_results.append(item)

    return fused_results


def hybrid_search(
    query: str,
    repo_name: str,
    top_k_retrieval: int = TOP_K_RETRIEVAL,
    top_k_final: int = TOP_K_FINAL,
    rrf_k: int = RRF_K_DEFAULT,
) -> list[dict[str, Any]]:
    """
    Perform hybrid search using Vector + BM25 with Reciprocal Rank Fusion.

    Retrieves top_k_retrieval candidates from Vector search and BM25 search,
    combines them with RRF, and returns the top_k_final results.

    Args:
        query: User search query.
        repo_name: Repository name.
        top_k_retrieval: How many chunks to retrieve per individual searcher (default: 20).
        top_k_final: Final number of fused chunks to return (default: 4).
        rrf_k: RRF smoothing constant (default: 60).

    Returns:
        List of fused result dicts of length <= top_k_final.
    """
    vector_results = vector_search(query, repo_name, top_k=top_k_retrieval)
    bm25_results = bm25_search(query, repo_name, top_k=top_k_retrieval)

    # Fuse both lists with RRF: index 0 = vector, index 1 = bm25
    fused = compute_rrf_fusion([vector_results, bm25_results], rrf_k=rrf_k)

    return fused[:top_k_final]


# ── Comparison Utility & CLI ─────────────────────────────────────────────────

def compare_search_modes(
    query: str,
    repo_name: str,
    top_k: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """
    Run Vector, BM25, and Hybrid search for comparison.

    Returns dict mapping mode names to their top-k result lists.
    """
    vec = vector_search(query, repo_name, top_k=top_k)
    bm = bm25_search(query, repo_name, top_k=top_k)
    hyb = hybrid_search(query, repo_name, top_k_retrieval=20, top_k_final=top_k)

    return {
        "vector": vec,
        "bm25": bm,
        "hybrid": hyb,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.retrieval.hybrid <repo_name> <query>")
        print('Example: python -m src.retrieval.hybrid flask-realworld-example-app "jwt_optional"')
        sys.exit(1)

    repo_name = sys.argv[1]
    query = " ".join(sys.argv[2:])

    print(f"\n======================================================================")
    print(f" 🔀 SEARCH MODE COMPARISON")
    print(f" Query: \"{query}\"")
    print(f" Repo:  {repo_name}")
    print(f"======================================================================")

    data = compare_search_modes(query, repo_name, top_k=5)

    def print_section(title: str, items: list[dict[str, Any]], score_label: str) -> None:
        print(f"\n--- {title} (Top {len(items)}) ---")
        if not items:
            print("  (No results)")
            return
        for item in items:
            sym = item["symbol_name"] or "(module/block)"
            parent = f" in {item['parent_class']}" if item["parent_class"] else ""
            cit = f"{item['file_path']}:{item['start_line']}-{item['end_line']}"
            extra = ""
            if "vector_rank" in item and "bm25_rank" in item:
                v_r = f"V#{item['vector_rank']}" if item['vector_rank'] else "V:—"
                b_r = f"B#{item['bm25_rank']}" if item['bm25_rank'] else "B:—"
                extra = f" [{v_r}, {b_r}]"
            print(f"  #{item['rank']:<2} {score_label}={item['score']:.4f}{extra} | {item['chunk_type']} '{sym}'{parent}")
            print(f"       📄 {cit}")

    print_section("1. VECTOR ONLY (Semantic Similarity)", data["vector"], "score")
    print_section("2. BM25 ONLY (Keyword Matching)", data["bm25"], "BM25")
    print_section("3. HYBRID (RRF Fusion)", data["hybrid"], "RRF")
    print(f"\n======================================================================\n")

