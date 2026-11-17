"""
Unit tests for Reranker, Caching, and Cloud Provider (Phase 8).

Tests:
  - Cross-Encoder reranking order, rank updating, and graceful fallback.
  - Repository fingerprinting and caching helpers.
  - Gemini LLM provider health check and payload formatting.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.retrieval.reranker import rerank, hybrid_search_with_rerank
from src.ingestion.indexer import (
    compute_repo_fingerprint,
    get_meta_path,
    is_repo_indexed,
)
from src.generation.llm import GeminiProvider, get_llm_provider


class TestReranker:
    """Tests for Cross-Encoder candidate re-ranking."""

    def test_rerank_empty_candidates(self):
        assert rerank("test query", [], top_k=4) == []

    def test_rerank_top_k_zero(self):
        candidates = [{"content": "x", "file_path": "a.py"}]
        assert rerank("test query", candidates, top_k=0) == []

    def test_rerank_fallback_when_model_none(self):
        candidates = [
            {"content": "first", "file_path": "a.py", "rank": 1},
            {"content": "second", "file_path": "b.py", "rank": 2},
        ]
        with patch("src.retrieval.reranker.get_reranker_model", return_value=None):
            result = rerank("query", candidates, top_k=2)
            assert len(result) == 2
            assert result[0]["file_path"] == "a.py"
            assert result[0]["rank"] == 1
            assert result[1]["file_path"] == "b.py"
            assert result[1]["rank"] == 2

    def test_rerank_scores_and_reorders(self):
        candidates = [
            {"content": "less relevant passage", "file_path": "a.py", "symbol_name": "foo"},
            {"content": "def login(): pass", "file_path": "b.py", "symbol_name": "login"},
        ]
        mock_model = MagicMock()
        # Mock predict returns higher score for candidate 1 than candidate 0
        mock_model.predict.return_value = [-2.5, 4.2]

        with patch("src.retrieval.reranker.get_reranker_model", return_value=mock_model):
            result = rerank("where is login?", candidates, top_k=2)

            assert len(result) == 2
            # Candidate with score 4.2 should be rank 1
            assert result[0]["file_path"] == "b.py"
            assert result[0]["rank"] == 1
            assert result[0]["rerank_score"] == 4.2

            # Candidate with score -2.5 should be rank 2
            assert result[1]["file_path"] == "a.py"
            assert result[1]["rank"] == 2
            assert result[1]["rerank_score"] == -2.5


class TestIndexerCaching:
    """Tests for repository content fingerprinting and caching."""

    def test_compute_repo_fingerprint_deterministic(self, tmp_path):
        f1 = tmp_path / "mod1.py"
        f1.write_text("print('hello')", encoding="utf-8")
        files = [
            {"path": str(f1), "relative_path": "mod1.py", "size_kb": 1.0},
        ]

        fp1 = compute_repo_fingerprint(files)
        fp2 = compute_repo_fingerprint(files)
        assert fp1 == fp2
        assert fp1.startswith("content:")

    def test_compute_repo_fingerprint_detects_change(self, tmp_path):
        f1 = tmp_path / "mod1.py"
        f1.write_text("v1", encoding="utf-8")
        files_v1 = [
            {"path": str(f1), "relative_path": "mod1.py", "size_kb": 1.0},
        ]
        fp1 = compute_repo_fingerprint(files_v1)

        # Change file size and content
        files_v2 = [
            {"path": str(f1), "relative_path": "mod1.py", "size_kb": 2.5},
        ]
        fp2 = compute_repo_fingerprint(files_v2)

        assert fp1 != fp2

    def test_get_meta_path(self):
        meta_path = get_meta_path("my-test-repo")
        assert meta_path.name == "my-test-repo_meta.json"

    def test_is_repo_indexed_for_nonexistent(self):
        assert is_repo_indexed("completely-nonexistent-repo-xyz") is False


class TestGeminiProvider:
    """Tests for the optional Google Gemini provider."""

    def test_unconfigured_api_key_fails_health_check(self):
        provider = GeminiProvider(api_key="")
        healthy, msg = provider.check_health()
        assert healthy is False
        assert "GEMINI_API_KEY is not configured" in msg

    def test_configured_api_key_passes_health_check(self):
        provider = GeminiProvider(api_key="AIzaSyDummyKeyForTesting12345")
        healthy, msg = provider.check_health()
        assert healthy is True
        assert "Gemini API configured" in msg

    def test_build_payload_structure(self):
        provider = GeminiProvider(api_key="dummy_key", model="gemini-1.5-flash", temperature=0.2)
        payload = provider._build_payload(
            prompt="Hello from user",
            system_prompt="You are a code assistant",
        )

        assert "contents" in payload
        assert payload["contents"][0]["parts"][0]["text"] == "Hello from user"
        assert "system_instruction" in payload
        assert payload["system_instruction"]["parts"][0]["text"] == "You are a code assistant"
        assert payload["generationConfig"]["temperature"] == 0.2

    def test_factory_returns_gemini_provider(self):
        provider = get_llm_provider(provider_name="gemini")
        assert isinstance(provider, GeminiProvider)

