"""Tests for Streamlit UI helpers (app.py)."""

from app import MODE_MAP, check_repo_indexed, resolve_repo_name


class TestUIHelpers:
    def test_resolve_repo_name_github_urls(self):
        assert resolve_repo_name("https://github.com/pallets/flask") == "flask"
        assert resolve_repo_name("https://github.com/pallets/flask.git") == "flask"
        assert resolve_repo_name("https://github.com/pallets/flask/") == "flask"

    def test_resolve_repo_name_local_paths(self):
        assert resolve_repo_name("data/flask-realworld-example-app") == "flask-realworld-example-app"
        assert resolve_repo_name("data\\flask-realworld-example-app") == "flask-realworld-example-app"
        assert resolve_repo_name("my_codebase") == "my_codebase"

    def test_check_repo_indexed(self):
        assert check_repo_indexed("flask-realworld-example-app") is True
        assert check_repo_indexed("nonexistent_repo_xyz_123") is False

    def test_mode_map_covers_all_four_modes(self):
        assert set(MODE_MAP.values()) == {"hybrid", "hybrid_rerank", "vector", "bm25"}
