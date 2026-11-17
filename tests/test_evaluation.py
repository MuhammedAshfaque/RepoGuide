"""
Unit tests for the evaluation framework (Phase 7).

Tests ground-truth matching (is_chunk_relevant), ranking metrics calculation
(Hit@k, MRR, latency), and report generation.
"""

import pytest
from pathlib import Path
from evaluation.run_eval import (
    is_chunk_relevant,
    evaluate_query_results,
    aggregate_metrics,
    generate_markdown_report,
)


class TestIsChunkRelevant:
    """Tests for ground truth relevance matching."""

    def test_exact_file_and_symbol_match(self):
        chunk = {
            "file_path": "conduit/user/models.py",
            "symbol_name": "User",
            "parent_class": None,
        }
        assert is_chunk_relevant(chunk, "conduit/user/models.py", "User") is True

    def test_case_and_slash_normalization(self):
        # Windows backslashes and different case
        chunk = {
            "file_path": "conduit\\User\\Models.py",
            "symbol_name": "user",
            "parent_class": None,
        }
        assert is_chunk_relevant(chunk, "conduit/user/models.py", "User") is True

    def test_relative_file_endswith_match(self):
        chunk = {
            "file_path": "repo_root/src/conduit/app.py",
            "symbol_name": "create_app",
            "parent_class": None,
        }
        assert is_chunk_relevant(chunk, "conduit/app.py", "create_app") is True

    def test_wrong_file_returns_false(self):
        chunk = {
            "file_path": "conduit/articles/views.py",
            "symbol_name": "login",
            "parent_class": None,
        }
        assert is_chunk_relevant(chunk, "conduit/user/views.py", "login") is False

    def test_empty_expected_symbol_matches_file(self):
        chunk = {
            "file_path": "conduit/settings.py",
            "symbol_name": "Config",
            "parent_class": None,
        }
        # When expected symbol is empty, any chunk from that file is relevant
        assert is_chunk_relevant(chunk, "conduit/settings.py", "") is True

    def test_enclosing_class_match(self):
        chunk = {
            "file_path": "conduit/user/models.py",
            "symbol_name": "check_password",
            "parent_class": "User",
        }
        # If question asks about User, the method under User should count as relevant
        assert is_chunk_relevant(chunk, "conduit/user/models.py", "User") is True

    def test_sub_symbol_match(self):
        chunk = {
            "file_path": "conduit/user/views.py",
            "symbol_name": "user_login_endpoint",
            "parent_class": None,
        }
        assert is_chunk_relevant(chunk, "conduit/user/views.py", "login") is True

    def test_same_file_different_symbol_returns_false(self):
        chunk = {
            "file_path": "conduit/user/models.py",
            "symbol_name": "some_other_helper",
            "parent_class": None,
        }
        assert is_chunk_relevant(chunk, "conduit/user/models.py", "User") is False


class TestEvaluateQueryResults:
    """Tests for single query evaluation against ground truth."""

    def test_first_rank_hit(self):
        results = [
            {"file_path": "conduit/app.py", "symbol_name": "create_app", "parent_class": None},
            {"file_path": "conduit/user/models.py", "symbol_name": "User", "parent_class": None},
        ]
        eval_result = evaluate_query_results(results, "conduit/app.py", "create_app")
        assert eval_result["first_rank"] == 1
        assert eval_result["reciprocal_rank"] == 1.0
        assert eval_result["hit@1"] == 1
        assert eval_result["hit@3"] == 1
        assert eval_result["hit@5"] == 1

    def test_rank_3_hit(self):
        results = [
            {"file_path": "unrelated1.py", "symbol_name": "foo", "parent_class": None},
            {"file_path": "unrelated2.py", "symbol_name": "bar", "parent_class": None},
            {"file_path": "conduit/app.py", "symbol_name": "create_app", "parent_class": None},
            {"file_path": "unrelated3.py", "symbol_name": "baz", "parent_class": None},
        ]
        eval_result = evaluate_query_results(results, "conduit/app.py", "create_app")
        assert eval_result["first_rank"] == 3
        assert pytest.approx(eval_result["reciprocal_rank"], 0.001) == 1.0 / 3.0
        assert eval_result["hit@1"] == 0
        assert eval_result["hit@3"] == 1
        assert eval_result["hit@5"] == 1

    def test_rank_5_hit(self):
        results = [
            {"file_path": f"dummy{i}.py", "symbol_name": "f", "parent_class": None}
            for i in range(4)
        ] + [{"file_path": "target.py", "symbol_name": "target_fn", "parent_class": None}]

        eval_result = evaluate_query_results(results, "target.py", "target_fn")
        assert eval_result["first_rank"] == 5
        assert pytest.approx(eval_result["reciprocal_rank"], 0.001) == 0.2
        assert eval_result["hit@1"] == 0
        assert eval_result["hit@3"] == 0
        assert eval_result["hit@5"] == 1

    def test_no_hit(self):
        results = [
            {"file_path": "unrelated.py", "symbol_name": "foo", "parent_class": None},
        ]
        eval_result = evaluate_query_results(results, "conduit/app.py", "create_app")
        assert eval_result["first_rank"] == 0
        assert eval_result["reciprocal_rank"] == 0.0
        assert eval_result["hit@1"] == 0
        assert eval_result["hit@3"] == 0
        assert eval_result["hit@5"] == 0


class TestAggregateMetrics:
    """Tests for aggregating query evaluations into dataset metrics."""

    def test_aggregate_empty(self):
        metrics = aggregate_metrics([], [])
        assert metrics["hit@1"] == 0.0
        assert metrics["hit@3"] == 0.0
        assert metrics["hit@5"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["latency_ms"] == 0.0

    def test_aggregate_known_values(self):
        evals = [
            {"hit@1": 1, "hit@3": 1, "hit@5": 1, "reciprocal_rank": 1.0},
            {"hit@1": 0, "hit@3": 1, "hit@5": 1, "reciprocal_rank": 0.5},
            {"hit@1": 0, "hit@3": 0, "hit@5": 1, "reciprocal_rank": 0.2},
            {"hit@1": 0, "hit@3": 0, "hit@5": 0, "reciprocal_rank": 0.0},
        ]
        latencies = [0.010, 0.020, 0.030, 0.040]  # in seconds

        metrics = aggregate_metrics(evals, latencies)
        # Hit@1 = 1/4 = 25.0%
        # Hit@3 = 2/4 = 50.0%
        # Hit@5 = 3/4 = 75.0%
        # MRR = (1.0 + 0.5 + 0.2 + 0.0) / 4 = 1.7 / 4 = 0.425
        # Latency = 0.025s = 25.0 ms
        assert metrics["hit@1"] == 25.0
        assert metrics["hit@3"] == 50.0
        assert metrics["hit@5"] == 75.0
        assert metrics["mrr"] == 0.4250
        assert metrics["latency_ms"] == 25.0


class TestGenerateMarkdownReport:
    """Tests for markdown evaluation report generation."""

    def test_report_generation(self, tmp_path):
        dummy_data = {
            "repo_name": "test-repo",
            "total_queries": 2,
            "overall": {
                "vector_only": {"hit@1": 50.0, "hit@3": 50.0, "hit@5": 100.0, "mrr": 0.6500, "latency_ms": 12.5},
                "bm25_only": {"hit@1": 50.0, "hit@3": 100.0, "hit@5": 100.0, "mrr": 0.7500, "latency_ms": 1.2},
                "hybrid": {"hit@1": 100.0, "hit@3": 100.0, "hit@5": 100.0, "mrr": 1.0000, "latency_ms": 13.8},
            },
            "breakdown": {
                "Conceptual / Natural Language": {
                    "vector_only": {"hit@5": 100.0, "mrr": 0.8},
                    "bm25_only": {"hit@5": 50.0, "mrr": 0.5},
                    "hybrid": {"hit@5": 100.0, "mrr": 1.0},
                },
                "Exact Symbol / Identifier": {
                    "vector_only": {"hit@5": 50.0, "mrr": 0.5},
                    "bm25_only": {"hit@5": 100.0, "mrr": 1.0},
                    "hybrid": {"hit@5": 100.0, "mrr": 1.0},
                },
            },
        }

        output_file = tmp_path / "results.md"
        content = generate_markdown_report(dummy_data, output_file)

        assert output_file.exists()
        assert "# CodeCompass — Retrieval Evaluation Report" in content
        assert "Hit@1" in content
        assert "Hit@5" in content
        assert "MRR (Mean Reciprocal Rank)" in content
        assert "Analysis & Observations" in content
        assert "100.0%" in content

