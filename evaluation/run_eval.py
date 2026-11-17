"""
Evaluation framework for CodeCompass (Phase 7).

Benchmarks retrieval performance across three search configurations:
  1. Vector Only (Dense semantic embeddings via all-MiniLM-L6-v2)
  2. BM25 Only (Sparse keyword matching with identifier tokenization)
  3. Hybrid (Reciprocal Rank Fusion of Vector + BM25)

Computes standard Information Retrieval (IR) metrics:
  - Hit@1, Hit@3, Hit@5: Recall at rank cutoff k.
  - MRR (Mean Reciprocal Rank): Measures ranking precision.
  - Average latency (ms) per query.

Generates `evaluation/results.md` with comparative analysis.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.retrieval.retrieval import get_retriever


# ── Ground Truth Evaluation Matching ─────────────────────────────────────────

def is_chunk_relevant(item: Document | dict[str, Any], expected_file: str, expected_symbol: str) -> bool:
    """
    Determine if a retrieved Document or chunk dict matches the ground-truth target.

    Args:
        item: Retrieved LangChain Document or dict containing 'file_path', 'symbol_name', 'parent_class'.
        expected_file: Ground truth relative file path.
        expected_symbol: Ground truth function, class, or method name.

    Returns:
        True if the chunk corresponds to the ground truth target.
    """
    meta = item.metadata if isinstance(item, Document) else item
    ret_file = str(meta.get("file_path", "")).replace("\\", "/").lower()
    exp_file = expected_file.replace("\\", "/").lower()

    file_matches = ret_file == exp_file or ret_file.endswith(exp_file)
    if not file_matches:
        return False

    if not expected_symbol:
        return True

    ret_sym = str(meta.get("symbol_name") or "").lower()
    exp_sym = expected_symbol.lower()
    ret_parent = str(meta.get("parent_class") or "").lower()

    # Exact symbol name match, sub-name match, or enclosing class match
    symbol_matches = (
        ret_sym == exp_sym
        or exp_sym in ret_sym
        or ret_parent == exp_sym
        or (ret_sym == "module" and exp_sym in ret_file)
    )

    return symbol_matches


def evaluate_query_results(
    results: list[Document] | list[dict[str, Any]],
    expected_file: str,
    expected_symbol: str,
    k_cutoffs: list[int] = [1, 3, 5],
) -> dict[str, Any]:
    """
    Evaluate a single query's ranked results against ground truth.

    Returns:
        Dict with first_hit_rank (or 0), reciprocal_rank, and hit booleans for each k.
    """
    first_rank = 0

    for idx, item in enumerate(results, 1):
        if is_chunk_relevant(item, expected_file, expected_symbol):
            first_rank = idx
            break

    reciprocal_rank = 1.0 / first_rank if first_rank > 0 else 0.0

    eval_dict: dict[str, Any] = {
        "first_rank": first_rank,
        "reciprocal_rank": reciprocal_rank,
    }

    for k in k_cutoffs:
        eval_dict[f"hit@{k}"] = 1 if (first_rank > 0 and first_rank <= k) else 0

    return eval_dict


# ── Metric Aggregator ────────────────────────────────────────────────────────

def aggregate_metrics(query_evals: list[dict[str, Any]], latencies: list[float]) -> dict[str, float]:
    """Calculate aggregate Hit@k, MRR, and average latency."""
    total = len(query_evals)
    if total == 0:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0, "latency_ms": 0.0}

    hit1 = sum(e["hit@1"] for e in query_evals) / total
    hit3 = sum(e["hit@3"] for e in query_evals) / total
    hit5 = sum(e["hit@5"] for e in query_evals) / total
    mrr = sum(e["reciprocal_rank"] for e in query_evals) / total
    avg_latency = (sum(latencies) / total) * 1000.0 if latencies else 0.0

    return {
        "hit@1": round(hit1 * 100.0, 1),
        "hit@3": round(hit3 * 100.0, 1),
        "hit@5": round(hit5 * 100.0, 1),
        "mrr": round(mrr, 4),
        "latency_ms": round(avg_latency, 1),
    }


# ── Evaluation Runner ────────────────────────────────────────────────────────

def run_evaluation(
    questions_path: Path,
    repo_name: str,
    top_k: int = 5,
    include_reranker: bool = True,
) -> dict[str, Any]:
    """
    Run the full comparative evaluation across Vector, BM25, Hybrid, and optionally Rerank.
    """
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    modes = ["vector_only", "bm25_only", "hybrid"]
    if include_reranker:
        modes.append("hybrid_rerank")

    per_mode_evals: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}
    per_mode_latencies: dict[str, list[float]] = {m: [] for m in modes}

    print(f"\n======================================================================")
    print(f" 📊 CodeCompass Retrieval Evaluation")
    print(f" Target Repo:       {repo_name}")
    print(f" Benchmark Queries: {len(questions)}")
    print(f" Top-K Cutoff:      {top_k}")
    print(f" Modes:             {', '.join(modes)}")
    print(f"======================================================================")

    # Build LangChain retrievers once
    vec_retriever = get_retriever(repo_name, mode="vector", top_k=top_k)
    bm25_retriever = get_retriever(repo_name, mode="bm25", top_k=top_k)
    hyb_retriever = get_retriever(repo_name, mode="hybrid", top_k=top_k, top_k_retrieval=20)
    rerank_retriever = (
        get_retriever(repo_name, mode="hybrid_rerank", top_k=top_k, top_k_retrieval=20)
        if include_reranker
        else None
    )

    # ── Warm-up Phase ────────────────────────────────────────────────────────
    # Run a warm-up query through each mode to load model weights and indexes into RAM.
    # This ensures timing measurements reflect steady-state retrieval latency,
    # rather than one-time cold-start model initialization.
    print("[eval] Warming up models and indexes...")
    vec_retriever.invoke("warmup query")
    bm25_retriever.invoke("warmup query")
    hyb_retriever.invoke("warmup query")
    if rerank_retriever is not None:
        rerank_retriever.invoke("warmup query")
    print("[eval] Warm-up complete. Starting benchmark runs...\n")

    for i, q in enumerate(questions, 1):
        query = q["question"]
        exp_file = q["expected_file"]
        exp_sym = q["expected_symbol"]

        # 1. Vector Search
        t0 = time.perf_counter()
        vec_results = vec_retriever.invoke(query)
        t_vec = time.perf_counter() - t0
        vec_eval = evaluate_query_results(vec_results, exp_file, exp_sym)
        per_mode_evals["vector_only"].append(vec_eval)
        per_mode_latencies["vector_only"].append(t_vec)

        # 2. BM25 Search
        t0 = time.perf_counter()
        bm_results = bm25_retriever.invoke(query)
        t_bm = time.perf_counter() - t0
        bm_eval = evaluate_query_results(bm_results, exp_file, exp_sym)
        per_mode_evals["bm25_only"].append(bm_eval)
        per_mode_latencies["bm25_only"].append(t_bm)

        # 3. Hybrid Search
        t0 = time.perf_counter()
        hyb_results = hyb_retriever.invoke(query)
        t_hyb = time.perf_counter() - t0
        hyb_eval = evaluate_query_results(hyb_results, exp_file, exp_sym)
        per_mode_evals["hybrid"].append(hyb_eval)
        per_mode_latencies["hybrid"].append(t_hyb)

        status_msg = (
            f" [{i:02d}/{len(questions):02d}] "
            f"V: rank {vec_eval['first_rank'] or '—':<2} | "
            f"B: rank {bm_eval['first_rank'] or '—':<2} | "
            f"H: rank {hyb_eval['first_rank'] or '—':<2}"
        )

        # 4. Hybrid + Rerank (optional)
        if rerank_retriever is not None:
            t0 = time.perf_counter()
            rerank_results = rerank_retriever.invoke(query)
            t_rr = time.perf_counter() - t0
            rr_eval = evaluate_query_results(rerank_results, exp_file, exp_sym)
            per_mode_evals["hybrid_rerank"].append(rr_eval)
            per_mode_latencies["hybrid_rerank"].append(t_rr)
            status_msg += f" | R: rank {rr_eval['first_rank'] or '—':<2}"

        status_msg += f" | {q['category']:<13} -> {query[:35]}..."
        print(status_msg)

    # Calculate overall metrics
    overall_summary = {
        m: aggregate_metrics(per_mode_evals[m], per_mode_latencies[m])
        for m in modes
    }

    # Calculate category breakdowns (conceptual vs exact_symbol)
    conceptual_indices = [
        idx for idx, q in enumerate(questions) if q.get("category") != "exact_symbol"
    ]
    symbol_indices = [
        idx for idx, q in enumerate(questions) if q.get("category") == "exact_symbol"
    ]

    breakdown = {}
    for group_name, indices in [("Conceptual / Natural Language", conceptual_indices), ("Exact Symbol / Identifier", symbol_indices)]:
        breakdown[group_name] = {}
        for m in modes:
            subset_evals = [per_mode_evals[m][idx] for idx in indices]
            subset_lats = [per_mode_latencies[m][idx] for idx in indices]
            breakdown[group_name][m] = aggregate_metrics(subset_evals, subset_lats)

    return {
        "repo_name": repo_name,
        "total_queries": len(questions),
        "overall": overall_summary,
        "breakdown": breakdown,
    }


# ── Markdown Report Generator ────────────────────────────────────────────────

def generate_markdown_report(report_data: dict[str, Any], output_path: Path) -> str:
    """Format evaluation numbers into evaluation/results.md."""
    overall = report_data["overall"]
    breakdown = report_data["breakdown"]
    total = report_data["total_queries"]
    repo = report_data["repo_name"]
    has_rerank = "hybrid_rerank" in overall

    header_cols = "| Metric | Vector Only | BM25 Only | Hybrid (RRF) |"
    sep_cols = "|---|:---:|:---:|:---:|"
    if has_rerank:
        header_cols += " Hybrid + Rerank |"
        sep_cols += ":---:|"

    def make_row(metric_name: str, key: str, is_pct: bool = True) -> str:
        mode_keys = ["vector_only", "bm25_only", "hybrid"] + (["hybrid_rerank"] if has_rerank else [])
        vals = [overall[m][key] for m in mode_keys]
        best = max(vals)
        suffix = "%" if is_pct else ""

        cells = []
        for v in vals:
            text = f"{v}{suffix}" if is_pct else f"{v:.4f}"
            cells.append(f"**{text}**" if v == best else text)

        return f"| **{metric_name}** | " + " | ".join(cells) + " |"

    def make_breakdown_row(section: dict[str, Any], metric_name: str, key: str, is_pct: bool = True) -> str:
        mode_keys = ["vector_only", "bm25_only", "hybrid"] + (["hybrid_rerank"] if has_rerank else [])
        vals = [section.get(m, {}).get(key, 0) for m in mode_keys]
        best = max(vals)
        suffix = "%" if is_pct else ""

        cells = []
        for v in vals:
            text = f"{v}{suffix}" if is_pct else f"{v:.4f}"
            cells.append(f"**{text}**" if v == best else text)

        return f"| **{metric_name}** | " + " | ".join(cells) + " |"

    row_hit1 = make_row("Hit@1", "hit@1")
    row_hit3 = make_row("Hit@3", "hit@3")
    row_hit5 = make_row("Hit@5", "hit@5")
    row_mrr = make_row("MRR (Mean Reciprocal Rank)", "mrr", is_pct=False)
    row_lat = f"| **Average Latency (ms)** | {overall['vector_only']['latency_ms']} ms | {overall['bm25_only']['latency_ms']} ms | {overall['hybrid']['latency_ms']} ms |"
    if has_rerank:
        row_lat += f" {overall['hybrid_rerank']['latency_ms']} ms |"

    # Category breakdown rows
    conc = breakdown.get("Conceptual / Natural Language", {})
    conc_hit5_row = make_breakdown_row(conc, "Hit@5", "hit@5", is_pct=True)
    conc_mrr_row = make_breakdown_row(conc, "MRR", "mrr", is_pct=False)

    sym = breakdown.get("Exact Symbol / Identifier", {})
    sym_hit5_row = make_breakdown_row(sym, "Hit@5", "hit@5", is_pct=True)
    sym_mrr_row = make_breakdown_row(sym, "MRR", "mrr", is_pct=False)

    configs = "Vector-Only, BM25-Only, Hybrid (RRF)"
    if has_rerank:
        configs += ", Hybrid + Cross-Encoder Rerank"

    v_h5, b_h5, h_h5 = overall["vector_only"]["hit@5"], overall["bm25_only"]["hit@5"], overall["hybrid"]["hit@5"]
    v_h1, h_h1 = overall["vector_only"]["hit@1"], overall["hybrid"]["hit@1"]
    v_mrr, h_mrr = overall["vector_only"]["mrr"], overall["hybrid"]["mrr"]

    hit5_cmp = "higher than" if h_h5 > v_h5 else ("equal to" if h_h5 == v_h5 else "lower than")
    mrr_cmp = "higher than" if h_mrr > v_mrr else ("equal to" if h_mrr == v_mrr else "lower than")

    if has_rerank:
        r_hit1 = overall["hybrid_rerank"]["hit@1"]
        r_hit5 = overall["hybrid_rerank"]["hit@5"]
        r_mrr = overall["hybrid_rerank"]["mrr"]
        r_lat = overall["hybrid_rerank"]["latency_ms"]
        rerank_bullets = f"""2. **Reranker Trade-offs:**
   - **Ranking Precision & Recall:** Hybrid + Rerank achieved Hit@1 of **{r_hit1}%**, Hit@5 of **{r_hit5}%**, and MRR of **{r_mrr:.4f}** (compared to Hybrid RRF at {h_h1}% Hit@1, {h_h5}% Hit@5, and {h_mrr:.4f} MRR).
   - **Latency:** Hybrid + Rerank had the highest average latency (**{r_lat} ms** vs. {overall['hybrid']['latency_ms']} ms for Hybrid, {overall['vector_only']['latency_ms']} ms for Vector, and {overall['bm25_only']['latency_ms']} ms for BM25)."""
    else:
        rerank_bullets = "2. **Reranker:** Not evaluated in this run."

    md = f"""# CodeCompass — Retrieval Evaluation Report

> **Target Repository:** `{repo}`  
> **Benchmark Dataset:** {total} ground-truth questions  
> **Retrieval Configurations:** {configs}  

---

## 1. Overall Performance Comparison

{header_cols}
{sep_cols}
{row_hit1}
{row_hit3}
{row_hit5}
{row_mrr}
{row_lat}

---

## 2. Category Breakdown Analysis

### A. Conceptual & Architectural Queries (Natural Language)
*Queries asking where components live or how flows work (e.g. "Where is the Flask application factory defined?", "How does password hashing work?").*

{header_cols}
{sep_cols}
{conc_hit5_row}
{conc_mrr_row}

### B. Exact Identifier & Symbol Queries
*Queries naming exact programming symbols (e.g. "`jwt_identity`", "`SurrogatePK`", "`is_following`").*

{header_cols}
{sep_cols}
{sym_hit5_row}
{sym_mrr_row}

---

## 3. Analysis & Observations

1. **Hybrid (RRF) vs. Vector Only and BM25 Only:**
   - **Hit@5:** Hybrid (RRF) reached **{h_h5}%**, which is {hit5_cmp} Vector Only ({v_h5}%) and higher than BM25 Only ({b_h5}%).
   - **Hit@1 and MRR:** Hybrid (RRF) reached {h_h1}% Hit@1 and {h_mrr:.4f} MRR, which is {mrr_cmp} Vector Only ({v_h1}% Hit@1, {v_mrr:.4f} MRR).

{rerank_bullets}

3. **Scope and Dataset Limits:**
   - These findings are based on a small evaluation set of {total} questions on a single repository (`{repo}`). Results on larger repositories or other programming languages may differ.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)

    return md


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval benchmark evaluation.")
    parser.add_argument("--repo", default="flask-realworld-example-app", help="Repository name.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K cutoff.")
    parser.add_argument(
        "--questions",
        default="evaluation/eval_questions.json",
        help="Path to questions JSON file.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results.md",
        help="Path for output markdown report.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranker evaluation.",
    )

    args = parser.parse_args()

    q_path = Path(args.questions)
    out_path = Path(args.output)

    results = run_evaluation(
        q_path, args.repo, top_k=args.top_k, include_reranker=not args.no_rerank
    )
    generate_markdown_report(results, out_path)

    print(f"\n✅ Benchmark Complete! Results written to: {out_path}")
    print(f"\nSummary Table:")
    modes = list(results["overall"].keys())
    header = f"| {'Metric':<8} | " + " | ".join(f"{m:^12}" for m in modes) + " |"
    sep = f"|{'-'*10}|" + "|".join(f"{'-'*14}" for _ in modes) + "|"
    print(header)
    print(sep)
    for metric in ["hit@1", "hit@3", "hit@5", "mrr"]:
        row = f"| {metric:<8} | "
        for m in modes:
            val = results["overall"][m][metric]
            suffix = "%" if "hit" in metric else ""
            row += f"{str(val) + suffix:>12} | "
        print(row)

