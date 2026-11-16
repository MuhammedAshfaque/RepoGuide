"""
Tests for the repository loader (src/ingestion/loader.py).

These tests verify:
  - URL parsing extracts the correct repo name
  - File filtering logic (extensions, size, skip dirs/files)
  - load_files returns the right structure
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.ingestion.loader import (
    repo_name_from_url,
    _should_skip_dir,
    _should_include_file,
    load_files,
)
from src.config import MAX_FILE_SIZE_KB


# ── URL parsing tests ────────────────────────────────────────────────────────

class TestRepoNameFromUrl:
    def test_basic_github_url(self):
        assert repo_name_from_url("https://github.com/user/my-repo") == "my-repo"

    def test_github_url_with_dot_git(self):
        assert repo_name_from_url("https://github.com/user/my-repo.git") == "my-repo"

    def test_github_url_with_trailing_slash(self):
        assert repo_name_from_url("https://github.com/user/my-repo/") == "my-repo"

    def test_github_url_with_dot_git_and_slash(self):
        assert repo_name_from_url("https://github.com/user/my-repo.git/") == "my-repo"


# ── Directory skip tests ────────────────────────────────────────────────────

class TestShouldSkipDir:
    def test_skip_git_dir(self):
        assert _should_skip_dir(".git") is True

    def test_skip_node_modules(self):
        assert _should_skip_dir("node_modules") is True

    def test_skip_pycache(self):
        assert _should_skip_dir("__pycache__") is True

    def test_skip_hidden_dir(self):
        assert _should_skip_dir(".hidden") is True

    def test_keep_normal_dir(self):
        assert _should_skip_dir("src") is False

    def test_keep_normal_dir_with_numbers(self):
        assert _should_skip_dir("api_v2") is False


# ── File inclusion tests ────────────────────────────────────────────────────

class TestShouldIncludeFile:
    """
    Uses a temporary directory to create real files so we can test
    file size checks, extension checks, and name-based skipping.
    """

    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path: Path):
        """Create a temp directory used as the fake repo root."""
        self.repo_root = tmp_path

    def _make_file(self, name: str, size_bytes: int = 10) -> Path:
        """Helper: create a file with a given size inside the temp dir."""
        fpath = self.repo_root / name
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(b"x" * size_bytes)
        return fpath

    def test_python_file_included(self):
        f = self._make_file("app.py")
        assert _should_include_file(f, self.repo_root) is True

    def test_js_file_included(self):
        f = self._make_file("index.js")
        assert _should_include_file(f, self.repo_root) is True

    def test_markdown_included(self):
        f = self._make_file("README.md")
        assert _should_include_file(f, self.repo_root) is True

    def test_binary_image_excluded(self):
        f = self._make_file("logo.png")
        assert _should_include_file(f, self.repo_root) is False

    def test_lock_file_excluded(self):
        f = self._make_file("package-lock.json")
        assert _should_include_file(f, self.repo_root) is False

    def test_unknown_extension_excluded(self):
        f = self._make_file("data.xyz")
        assert _should_include_file(f, self.repo_root) is False

    def test_oversized_file_excluded(self):
        # Create a file just over the MAX_FILE_SIZE_KB limit
        size = (MAX_FILE_SIZE_KB + 1) * 1024
        f = self._make_file("big_file.py", size_bytes=size)
        assert _should_include_file(f, self.repo_root) is False

    def test_file_just_under_limit_included(self):
        size = (MAX_FILE_SIZE_KB - 1) * 1024
        f = self._make_file("ok_file.py", size_bytes=size)
        assert _should_include_file(f, self.repo_root) is True

    def test_no_extension_file_included(self):
        """Files like Makefile, Dockerfile with no extension should be kept."""
        f = self._make_file("Makefile")
        assert _should_include_file(f, self.repo_root) is True


# ── Integration: load_files on a fake local directory ────────────────────────

class TestLoadFilesLocal:
    """
    Build a small fake repo structure in a temp dir and verify
    that load_files returns the expected files with correct metadata.
    """

    @pytest.fixture(autouse=True)
    def setup_fake_repo(self, tmp_path: Path):
        """Create a mini fake repo structure."""
        self.repo_root = tmp_path / "fake_repo"
        self.repo_root.mkdir()

        # Good files (should be included)
        (self.repo_root / "app.py").write_text("print('hello')")
        (self.repo_root / "README.md").write_text("# Readme")

        sub = self.repo_root / "src"
        sub.mkdir()
        (sub / "utils.py").write_text("def helper(): pass")

        # Junk that should be filtered out
        git_dir = self.repo_root / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("gitconfig")

        nm = self.repo_root / "node_modules"
        nm.mkdir()
        (nm / "lodash.js").write_text("module.exports = {}")

        (self.repo_root / "logo.png").write_bytes(b"\x89PNG" + b"\x00" * 100)

    def test_returns_correct_file_count(self):
        results = load_files(str(self.repo_root))
        # Should find: app.py, README.md, src/utils.py = 3 files
        assert len(results) == 3

    def test_result_has_expected_keys(self):
        results = load_files(str(self.repo_root))
        for r in results:
            assert "path" in r
            assert "relative_path" in r
            assert "extension" in r
            assert "size_kb" in r

    def test_relative_paths_are_correct(self):
        results = load_files(str(self.repo_root))
        rel_paths = {r["relative_path"] for r in results}
        assert "app.py" in rel_paths
        assert "README.md" in rel_paths
        # os.path.join uses backslash on Windows
        assert "src\\utils.py" in rel_paths or "src/utils.py" in rel_paths

    def test_no_git_files_included(self):
        results = load_files(str(self.repo_root))
        for r in results:
            assert ".git" not in r["relative_path"]

    def test_no_node_modules_included(self):
        results = load_files(str(self.repo_root))
        for r in results:
            assert "node_modules" not in r["relative_path"]

    def test_no_binary_files_included(self):
        results = load_files(str(self.repo_root))
        for r in results:
            assert not r["relative_path"].endswith(".png")

