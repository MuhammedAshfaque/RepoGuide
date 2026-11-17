"""
Unit tests for the Streamlit UI helper functions (Phase 6).

Covers:
  - Repository name resolution from URLs and local paths
  - Index presence checking
  - Retrieval mode routing
"""

from pathlib import Path
import pytest

from app import resolve_repo_name, check_repo_indexed, retrieve_by_mode


class TestUIHelpers:
    def test_resolve_repo_name_github_urls(self):
        assert resolve_repo_name("https://github.com/pallets/flask") == "flask"
        assert resolve_repo_name("https://github.com/pallets/flask.git") == "flask"
        assert resolve_repo_name("https://github.com/pallets/flask/") == "flask"

    def test_resolve_repo_name_local_paths(self):
        assert resolve_repo_name("data/flask-realworld-example-app") == "flask-realworld-example-app"
        assert resolve_repo_name("data\\flask-realworld-example-app") == "flask-realworld-example-app"
        assert resolve_repo_name("my_codebase") == "my_codebase"

    def test_check_repo_indexed_for_existing_demo(self):
        # Demo repo was indexed in earlier phases
        assert check_repo_indexed("flask-realworld-example-app") is True

    def test_check_repo_indexed_for_nonexistent(self):
        assert check_repo_indexed("nonexistent_repo_xyz_123") is False

    def test_retrieve_by_mode_routes_to_correct_engine(self, monkeypatch):
        called = {}

        def mock_hybrid(q, r, top_k_retrieval=20, top_k_final=4):
            called["hybrid"] = (q, r, top_k_final)
            return [{"mode": "hybrid"}]

        def mock_vector(q, r, top_k=4):
            called["vector"] = (q, r, top_k)
            return [{"mode": "vector"}]

        def mock_bm25(q, r, top_k=4):
            called["bm25"] = (q, r, top_k)
            return [{"mode": "bm25"}]

        monkeypatch.setattr("app.hybrid_search", mock_hybrid)
        monkeypatch.setattr("app.vector_search", mock_vector)
        monkeypatch.setattr("app.bm25_search", mock_bm25)

        # 1. Hybrid route
        res_h = retrieve_by_mode("auth", "demo", "Hybrid (Vector + BM25 via RRF)", top_k=3)
        assert res_h[0]["mode"] == "hybrid"
        assert called["hybrid"] == ("auth", "demo", 3)

        # 2. Vector route
        res_v = retrieve_by_mode("auth", "demo", "Vector Only", top_k=5)
        assert res_v[0]["mode"] == "vector"
        assert called["vector"] == ("auth", "demo", 5)

        # 3. BM25 route
        res_b = retrieve_by_mode("auth", "demo", "BM25 Only", top_k=2)
        assert res_b[0]["mode"] == "bm25"
        assert called["bm25"] == ("auth", "demo", 2)

