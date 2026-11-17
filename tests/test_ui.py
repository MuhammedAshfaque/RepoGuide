"""
Unit tests for the Streamlit UI helper functions (app.py).

Covers:
  - Repository name resolution from URLs and local paths
  - Index presence checking (pickle docs file)
  - Retrieval mode routing to LangChain get_retriever()
"""

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app import check_repo_indexed, resolve_repo_name, retrieve_by_mode


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
        assert check_repo_indexed("flask-realworld-example-app") is True

    def test_check_repo_indexed_for_nonexistent(self):
        assert check_repo_indexed("nonexistent_repo_xyz_123") is False

    def test_retrieve_by_mode_routes_all_modes(self, monkeypatch):
        calls = []

        def mock_get_retriever(repo_name, mode="hybrid", top_k=4):
            calls.append((repo_name, mode, top_k))
            return RunnableLambda(lambda q: [Document(page_content=q, metadata={"mode": mode})])

        monkeypatch.setattr("app.get_retriever", mock_get_retriever)

        res_h = retrieve_by_mode("auth", "demo", "Hybrid (Vector + BM25 via RRF)", top_k=3)
        res_r = retrieve_by_mode("auth", "demo", "Hybrid + Rerank", top_k=4)
        res_v = retrieve_by_mode("auth", "demo", "Vector Only", top_k=2)
        res_b = retrieve_by_mode("auth", "demo", "BM25 Only", top_k=1)

        assert res_h[0].metadata["mode"] == "hybrid"
        assert res_r[0].metadata["mode"] == "hybrid_rerank"
        assert res_v[0].metadata["mode"] == "vector"
        assert res_b[0].metadata["mode"] == "bm25"
        assert calls == [
            ("demo", "hybrid", 3),
            ("demo", "hybrid_rerank", 4),
            ("demo", "vector", 2),
            ("demo", "bm25", 1),
        ]
