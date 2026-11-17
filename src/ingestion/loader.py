"""Repository loader: shallow-clones GitHub URLs or scans local folders with file filtering."""

import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import git

from src.config import (
    ALLOWED_EXTENSIONS,
    BINARY_EXTENSIONS,
    DATA_DIR,
    MAX_FILE_SIZE_KB,
    SKIP_DIRS,
    SKIP_FILES,
)

_SKIP_DIRS_LOWER = {d.lower() for d in SKIP_DIRS}


def repo_name_from_url(url: str) -> str:
    """Extract the repository name from a GitHub URL."""
    name = urlparse(url).path.rstrip("/").split("/")[-1]
    return name[:-4] if name.endswith(".git") else name


def clone_repo(url: str, force_reclone: bool = False) -> Path:
    """Shallow-clone a GitHub repository into data/<repo_name>/."""
    dest = DATA_DIR / repo_name_from_url(url)
    if dest.exists():
        if force_reclone:
            shutil.rmtree(dest)
        else:
            print(f"[loader] Repo already cloned at {dest}. Skipping clone.")
            return dest

    print(f"[loader] Cloning {url} into {dest} ...")
    git.Repo.clone_from(url, str(dest), depth=1, single_branch=True)
    return dest


def resolve_repo_path(source: str, force_reclone: bool = False) -> Path:
    """Resolve a GitHub URL or local directory path to a folder on disk."""
    if source.startswith(("http://", "https://")):
        return clone_repo(source, force_reclone=force_reclone)
    local = Path(source)
    if local.is_dir():
        return local
    raise ValueError(f"Source must be a GitHub URL or existing directory. Got: {source!r}")


def _should_skip_dir(dir_name: str) -> bool:
    """Return True if a directory should be pruned during traversal."""
    if dir_name.startswith(".") and dir_name != ".":
        return True
    return dir_name.lower() in _SKIP_DIRS_LOWER


def _should_include_file(file_path: Path) -> bool:
    """Return True if a file passes name, extension allowlist, and size filters."""
    if file_path.name in SKIP_FILES:
        return False

    ext = file_path.suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return False
    # Allowlist: reject unknown extensions, but allow extensionless files (Makefile, Dockerfile)
    if ext and ext not in ALLOWED_EXTENSIONS:
        return False

    try:
        return (file_path.stat().st_size / 1024) <= MAX_FILE_SIZE_KB
    except OSError:
        return False


def load_files(source: str, force_reclone: bool = False) -> list[dict]:
    """Walk a repository and return metadata dicts for all indexable files."""
    repo_root = resolve_repo_path(source, force_reclone=force_reclone)
    print(f"[loader] Scanning files in {repo_root} ...")

    files: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            if _should_include_file(fpath):
                files.append({
                    "path": fpath,
                    "relative_path": str(fpath.relative_to(repo_root)),
                    "extension": fpath.suffix.lower(),
                    "size_kb": round(fpath.stat().st_size / 1024, 1),
                })

    print(f"[loader] Found {len(files)} indexable files.")
    return files
