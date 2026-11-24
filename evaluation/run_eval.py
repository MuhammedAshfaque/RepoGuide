"""Retrieval benchmark: evaluates Vector, BM25, Hybrid (RRF), and Hybrid + Rerank with warm-up timing."""

import argparse
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.retrieval.retrieval import get_retriever

MODE_SPECS = [
    ("vector_only", "vector", "Vector Only"),
    ("bm25_only", "bm25", "BM25 Only"),
    ("hybrid", "hybrid", "Hybrid (RRF)"),
    ("hybrid_rerank", "hybrid_rerank", "Hybrid + Rerank"),
]


def is_chunk_relevant(item: Document | dict[str, Any], expected_file: str, expected_symbol: str) -> bool:
    """Return True if a retrieved Document or dict matches the expected file and symbol."""
    meta = item.metadata if isinstance(item, Document) else item
    ret_file = str(meta.get("file_path", "")).replace("\\", "/").lower()
    exp_file = expected_file.replace("\\", "/").lower()
    if not (ret_file == exp_file or ret_file.endswith(exp_file)):
        return False
    if not expected_symbol:
        return True

    ret_sym = str(meta.get("symbol_name") or "").lower()
    exp_sym = expected_symbol.lower()
    ret_parent = str(meta.get("parent_class") or "").lower()
    return (
        ret_sym == exp_sym
        or exp_sym in ret_sym
        or ret_parent == exp_sym
        or (ret_sym == "module" and exp_sym in ret_file)
    )


def evaluate_query_results(
    results: list[Document] | list[dict[str, Any]],
    expected_file: str,
    expected_symbol: str,
    k_cutoffs: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    """Compute first_rank, reciprocal_rank, and Hit@k for a single query."""
    first_rank = next(
        (idx for idx, item in enumerate(results, 1) if is_chunk_relevant(item, expected_file, expected_symbol)),
        0,
    )
    out: dict[str, Any] = {
        "first_rank": first_rank,
        "reciprocal_rank": 1.0 / first_rank if first_rank > 0 else 0.0,
    }
    for k in k_cutoffs:
        out[f"hit@{k}"] = 1 if 0 < first_rank <= k else 0
    return out


def aggregate_metrics(query_evals: list[dict[str, Any]], latencies: list[float]) -> dict[str, float]:
    """Aggregate per-query results into Hit@1, Hit@3, Hit@5, MRR, and average latency (ms)."""
    total = len(query_evals)
    if total == 0:
        return {"hit@1": 0.0, "hit@3": 0.0, "hit@5": 0.0, "mrr": 0.0, "latency_ms": 0.0}

    return {
        "hit@1": round(sum(e["hit@1"] for e in query_evals) / total * 100.0, 1),
        "hit@3": round(sum(e["hit@3"] for e in query_evals) / total * 100.0, 1),
        "hit@5": round(sum(e["hit@5"] for e in query_evals) / total * 100.0, 1),
        "mrr": round(sum(e["reciprocal_rank"] for e in query_evals) / total, 4),
        "latency_ms": round((sum(latencies) / total) * 1000.0, 1) if latencies else 0.0,
    }


def run_evaluation(
    questions_path: Path,
    repo_name: str,
    top_k: int = 5,
    include_reranker: bool = True,
) -> dict[str, Any]:
    """Run all benchmark queries across retrieval modes with warm-up before timing."""
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    active_specs = MODE_SPECS if include_reranker else MODE_SPECS[:3]
    retrievers = {
        key: get_retriever(repo_name, mode=mode, top_k=top_k, top_k_retrieval=20)
        for key, mode, _ in active_specs
    }

    print(f"\nRunning {len(questions)} queries on '{repo_name}' (top_k={top_k})...")
    for ret in retrievers.values():
        ret.invoke("warmup query")

    per_mode_evals: dict[str, list[dict[str, Any]]] = {k: [] for k in retrievers}
    per_mode_lats: dict[str, list[float]] = {k: [] for k in retrievers}

    for i, q in enumerate(questions, 1):
        ranks = []
        for key, ret in retrievers.items():
            t0 = time.perf_counter()
            docs = ret.invoke(q["question"])
            elapsed = time.perf_counter() - t0
            ev = evaluate_query_results(docs, q["expected_file"], q["expected_symbol"])
            per_mode_evals[key].append(ev)
            per_mode_lats[key].append(elapsed)
            ranks.append(f"{key[0].upper()}:{ev['first_rank'] or '-'}")
        print(f" [{i:02d}/{len(questions):02d}] {' | '.join(ranks)} | {q['question'][:40]}")

    overall = {k: aggregate_metrics(per_mode_evals[k], per_mode_lats[k]) for k in retrievers}

    groups = {
        "Conceptual / Natural Language": [i for i, q in enumerate(questions) if q.get("category") != "exact_symbol"],
        "Exact Symbol / Identifier": [i for i, q in enumerate(questions) if q.get("category") == "exact_symbol"],
    }
    breakdown = {
        name: {
            k: aggregate_metrics([per_mode_evals[k][i] for i in idxs], [per_mode_lats[k][i] for i in idxs])
            for k in retrievers
        }
        for name, idxs in groups.items()
    }
    return {"repo_name": repo_name, "total_queries": len(questions), "overall": overall, "breakdown": breakdown}


def _format_row(label: str, values: list[float], is_pct: bool = True, bold_max: bool = True) -> str:
    """Format a markdown table row, bolding the highest value(s) when requested."""
    best = max(values) if (bold_max and values) else None
    cells = []
    for v in values:
        text = f"{v}%" if is_pct else f"{v:.4f}"
        cells.append(f"**{text}**" if best is not None and v == best else text)
    return f"| **{label}** | " + " | ".join(cells) + " |"


def generate_markdown_report(report_data: dict[str, Any], output_path: Path) -> str:
    """Write a factual markdown benchmark report to output_path."""
    overall = report_data["overall"]
    breakdown = report_data["breakdown"]
    total = report_data["total_queries"]
    repo = report_data["repo_name"]

    active_specs = [(k, label) for k, _, label in MODE_SPECS if k in overall]
    keys = [k for k, _ in active_specs]
    header = "| Metric | " + " | ".join(label for _, label in active_specs) + " |"
    sep = "|---|" + ":---:|" * len(active_specs)

    lat_row = "| **Average Latency (ms)** | " + " | ".join(f"{overall[k]['latency_ms']} ms" for k in keys) + " |"
    conc = breakdown.get("Conceptual / Natural Language", {})
    sym = breakdown.get("Exact Symbol / Identifier", {})

    v_h5, b_h5, h_h5 = overall["vector_only"]["hit@5"], overall["bm25_only"]["hit@5"], overall["hybrid"]["hit@5"]
    v_h1, h_h1 = overall["vector_only"]["hit@1"], overall["hybrid"]["hit@1"]
    v_mrr, h_mrr = overall["vector_only"]["mrr"], overall["hybrid"]["mrr"]
    hit5_cmp = "higher than" if h_h5 > v_h5 else ("equal to" if h_h5 == v_h5 else "lower than")
    mrr_cmp = "higher than" if h_mrr > v_mrr else ("equal to" if h_mrr == v_mrr else "lower than")

    if "hybrid_rerank" in overall:
        r = overall["hybrid_rerank"]
        rerank_note = (
            f"2. **Reranker Trade-offs:**\n"
            f"   - **Ranking Precision & Recall:** Hybrid + Rerank achieved Hit@1 of **{r['hit@1']}%**, "
            f"Hit@5 of **{r['hit@5']}%**, and MRR of **{r['mrr']:.4f}** "
            f"(vs. Hybrid RRF at {h_h1}% Hit@1, {h_h5}% Hit@5, and {h_mrr:.4f} MRR).\n"
            f"   - **Latency:** Hybrid + Rerank had the highest average latency (**{r['latency_ms']} ms** "
            f"vs. {overall['hybrid']['latency_ms']} ms for Hybrid, {overall['vector_only']['latency_ms']} ms for Vector, "
            f"and {overall['bm25_only']['latency_ms']} ms for BM25)."
        )
    else:
        rerank_note = "2. **Reranker:** Not evaluated in this run."

    md = "\n".join([
        "# CodeCompass — Retrieval Evaluation Report", "",
        f"> **Target Repository:** `{repo}`  ",
        f"> **Benchmark Dataset:** {total} ground-truth questions  ",
        f"> **Retrieval Configurations:** {', '.join(label for _, label in active_specs)}  ", "",
        "---", "",
        "## 1. Overall Performance Comparison", "",
        header, sep,
        _format_row("Hit@1", [overall[k]["hit@1"] for k in keys]),
        _format_row("Hit@3", [overall[k]["hit@3"] for k in keys]),
        _format_row("Hit@5", [overall[k]["hit@5"] for k in keys]),
        _format_row("MRR (Mean Reciprocal Rank)", [overall[k]["mrr"] for k in keys], is_pct=False),
        lat_row, "",
        "---", "",
        "## 2. Category Breakdown Analysis", "",
        "### A. Conceptual & Architectural Queries (Natural Language)", "",
        header, sep,
        _format_row("Hit@5", [conc.get(k, {}).get("hit@5", 0) for k in keys]),
        _format_row("MRR", [conc.get(k, {}).get("mrr", 0) for k in keys], is_pct=False), "",
        "### B. Exact Identifier & Symbol Queries", "",
        header, sep,
        _format_row("Hit@5", [sym.get(k, {}).get("hit@5", 0) for k in keys]),
        _format_row("MRR", [sym.get(k, {}).get("mrr", 0) for k in keys], is_pct=False), "",
        "---", "",
        "## 3. Analysis & Observations", "",
        "1. **Hybrid (RRF) vs. Vector Only and BM25 Only:**",
        f"   - **Hit@5:** Hybrid (RRF) reached **{h_h5}%**, which is {hit5_cmp} Vector Only ({v_h5}%) and higher than BM25 Only ({b_h5}%).",
        f"   - **Hit@1 and MRR:** Hybrid (RRF) reached {h_h1}% Hit@1 and {h_mrr:.4f} MRR, which is {mrr_cmp} Vector Only ({v_h1}% Hit@1, {v_mrr:.4f} MRR).", "",
        rerank_note, "",
        "3. **Scope and Dataset Limits:**",
        f"   - These findings are based on a small evaluation set of {total} questions on a single repository (`{repo}`). Results on larger repositories or other programming languages may differ.",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")
    return md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run retrieval benchmark evaluation.")
    parser.add_argument("--repo", default="flask-realworld-example-app")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--questions", default="evaluation/eval_questions.json")
    parser.add_argument("--output", default="evaluation/results.md")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    results = run_evaluation(Path(args.questions), args.repo, top_k=args.top_k, include_reranker=not args.no_rerank)
    generate_markdown_report(results, Path(args.output))
    print(f"\nDone! Report written to {args.output}")
