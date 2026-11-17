"""Tests for the repository loader (src/ingestion/loader.py)."""

from pathlib import Path
import pytest

from src.config import MAX_FILE_SIZE_KB
from src.ingestion.loader import (
    _should_include_file,
    _should_skip_dir,
    load_files,
    repo_name_from_url,
)


class TestRepoNameFromUrl:
    def test_basic_github_url(self):
        assert repo_name_from_url("https://github.com/user/my-repo") == "my-repo"

    def test_github_url_with_dot_git(self):
        assert repo_name_from_url("https://github.com/user/my-repo.git") == "my-repo"

    def test_github_url_with_trailing_slash(self):
        assert repo_name_from_url("https://github.com/user/my-repo/") == "my-repo"

    def test_github_url_with_dot_git_and_slash(self):
        assert repo_name_from_url("https://github.com/user/my-repo.git/") == "my-repo"


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


class TestShouldIncludeFile:
    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path: Path):
        self.repo_root = tmp_path

    def _make_file(self, name: str, size_bytes: int = 10) -> Path:
        fpath = self.repo_root / name
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(b"x" * size_bytes)
        return fpath

    def test_python_file_included(self):
        assert _should_include_file(self._make_file("app.py")) is True

    def test_js_file_included(self):
        assert _should_include_file(self._make_file("index.js")) is True

    def test_markdown_included(self):
        assert _should_include_file(self._make_file("README.md")) is True

    def test_binary_image_excluded(self):
        assert _should_include_file(self._make_file("logo.png")) is False

    def test_lock_file_excluded(self):
        assert _should_include_file(self._make_file("package-lock.json")) is False

    def test_unknown_extension_excluded(self):
        assert _should_include_file(self._make_file("data.xyz")) is False

    def test_oversized_file_excluded(self):
        f = self._make_file("big_file.py", size_bytes=(MAX_FILE_SIZE_KB + 1) * 1024)
        assert _should_include_file(f) is False

    def test_file_just_under_limit_included(self):
        f = self._make_file("ok_file.py", size_bytes=(MAX_FILE_SIZE_KB - 1) * 1024)
        assert _should_include_file(f) is True

    def test_no_extension_file_included(self):
        assert _should_include_file(self._make_file("Makefile")) is True


class TestLoadFilesLocal:
    @pytest.fixture(autouse=True)
    def setup_fake_repo(self, tmp_path: Path):
        self.repo_root = tmp_path / "fake_repo"
        self.repo_root.mkdir()
        (self.repo_root / "app.py").write_text("print('hello')")
        (self.repo_root / "README.md").write_text("# Readme")

        sub = self.repo_root / "src"
        sub.mkdir()
        (sub / "utils.py").write_text("def helper(): pass")

        for junk_dir, fname, text in [(".git", "config", "gitconfig"), ("node_modules", "lodash.js", "{}")]:
            d = self.repo_root / junk_dir
            d.mkdir()
            (d / fname).write_text(text)

        (self.repo_root / "logo.png").write_bytes(b"\x89PNG" + b"\x00" * 100)

    def test_returns_correct_file_count(self):
        assert len(load_files(str(self.repo_root))) == 3

    def test_result_has_expected_keys(self):
        for r in load_files(str(self.repo_root)):
            assert {"path", "relative_path", "extension", "size_kb"} <= set(r.keys())

    def test_relative_paths_are_correct(self):
        rel_paths = {r["relative_path"] for r in load_files(str(self.repo_root))}
        assert "app.py" in rel_paths
        assert "README.md" in rel_paths
        assert "src\\utils.py" in rel_paths or "src/utils.py" in rel_paths

    def test_no_git_files_included(self):
        assert all(".git" not in r["relative_path"] for r in load_files(str(self.repo_root)))

    def test_no_node_modules_included(self):
        assert all("node_modules" not in r["relative_path"] for r in load_files(str(self.repo_root)))

    def test_no_binary_files_included(self):
        assert all(not r["relative_path"].endswith(".png") for r in load_files(str(self.repo_root)))
