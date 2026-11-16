"""
Repository loader for CodeCompass.

Handles two input types:
  1. A GitHub URL  → clones the repo into data/<repo_name>/
  2. A local folder path → uses it directly

After resolving the repo path, it walks the directory tree and returns
a list of file paths that pass our filtering rules (skip junk dirs,
binary files, lock files, and anything over MAX_FILE_SIZE_KB).
"""

import os
import shutil
from pathlib import Path
from urllib.parse import urlparse

import git  # GitPython

from src.config import (
    DATA_DIR,
    SKIP_DIRS,
    ALLOWED_EXTENSIONS,
    BINARY_EXTENSIONS,
    SKIP_FILES,
    MAX_FILE_SIZE_KB,
)


def repo_name_from_url(url: str) -> str:
    """
    Extract a clean repo name from a GitHub URL.

    Examples:
        'https://github.com/user/repo'       → 'repo'
        'https://github.com/user/repo.git'   → 'repo'
    """
    parsed = urlparse(url)
    # The path looks like '/user/repo' or '/user/repo.git'
    name = parsed.path.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def clone_repo(url: str, force_reclone: bool = False) -> Path:
    """
    Clone a GitHub repo into data/<repo_name>/.

    Args:
        url: Full GitHub URL (https://github.com/user/repo).
        force_reclone: If True, delete existing clone and re-clone.

    Returns:
        Path to the cloned repository root.
    """
    name = repo_name_from_url(url)
    dest = DATA_DIR / name

    if dest.exists():
        if force_reclone:
            shutil.rmtree(dest)
        else:
            print(f"[loader] Repo already cloned at {dest}. Skipping clone.")
            return dest

    print(f"[loader] Cloning {url} into {dest} ...")
    git.Repo.clone_from(
        url,
        str(dest),
        depth=1,           # shallow clone — saves disk & time
        single_branch=True,
    )
    print(f"[loader] Clone complete.")
    return dest


def resolve_repo_path(source: str, force_reclone: bool = False) -> Path:
    """
    Accept either a GitHub URL or a local directory path.

    Returns the Path to the repo root on disk.

    Raises:
        ValueError: If the source is neither a valid URL nor an existing
                    local directory.
    """
    # If it looks like a URL, clone it
    if source.startswith("http://") or source.startswith("https://"):
        return clone_repo(source, force_reclone=force_reclone)

    # Otherwise treat it as a local path
    local = Path(source)
    if local.is_dir():
        return local

    raise ValueError(
        f"Source must be a GitHub URL or an existing local directory. "
        f"Got: {source!r}"
    )


def _should_skip_dir(dir_name: str) -> bool:
    """Return True if this directory name should be excluded."""
    # Skip hidden directories (except the repo root itself)
    if dir_name.startswith(".") and dir_name != ".":
        return True
    # Skip known junk directories
    if dir_name.lower() in {d.lower() for d in SKIP_DIRS}:
        return True
    return False


def _should_include_file(file_path: Path, repo_root: Path) -> bool:
    """
    Decide whether a file should be included for chunking/indexing.

    Rules (in order):
      1. Skip files in SKIP_FILES by exact name.
      2. Skip binary extensions.
      3. Only include files with an allowed extension (allowlist approach).
         Exception: files with NO extension (like Makefile, Dockerfile,
         Procfile) are included if they are small enough.
      4. Skip files larger than MAX_FILE_SIZE_KB.
    """
    name = file_path.name

    # Rule 1: skip known junk files
    if name in SKIP_FILES:
        return False

    # Rule 2: skip binary extensions
    ext = file_path.suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return False

    # Rule 3: allowlist check
    if ext and ext not in ALLOWED_EXTENSIONS:
        return False

    # Rule 4: size limit (in kilobytes)
    try:
        size_kb = file_path.stat().st_size / 1024
        if size_kb > MAX_FILE_SIZE_KB:
            return False
    except OSError:
        return False

    return True


def load_files(source: str, force_reclone: bool = False) -> list[dict]:
    """
    Main entry point: resolve repo, walk it, return filtered file info.

    Args:
        source: A GitHub URL or local directory path.
        force_reclone: If True and source is a URL, delete and re-clone.

    Returns:
        A list of dicts, each with:
            - 'path':          absolute Path object
            - 'relative_path': path relative to repo root (for display / citations)
            - 'extension':     file extension (e.g. '.py')
            - 'size_kb':       file size in KB (rounded to 1 decimal)

    Design note:
        We use an allowlist of extensions rather than a blocklist.
        This is safer — we only index files we know how to chunk,
        and we never accidentally ingest a 50 MB binary.
    """
    repo_root = resolve_repo_path(source, force_reclone=force_reclone)
    print(f"[loader] Scanning files in {repo_root} ...")

    files: list[dict] = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Modify dirnames in-place to prune entire subtrees
        dirnames[:] = [
            d for d in dirnames if not _should_skip_dir(d)
        ]

        for fname in filenames:
            fpath = Path(dirpath) / fname
            if _should_include_file(fpath, repo_root):
                files.append({
                    "path": fpath,
                    "relative_path": str(fpath.relative_to(repo_root)),
                    "extension": fpath.suffix.lower(),
                    "size_kb": round(fpath.stat().st_size / 1024, 1),
                })

    print(f"[loader] Found {len(files)} indexable files.")
    return files


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.loader <github_url_or_local_path>")
        sys.exit(1)

    source = sys.argv[1]
    results = load_files(source)

    # Print summary
    print(f"\n{'='*60}")
    print(f" Total files: {len(results)}")
    print(f"{'='*60}")

    # Group by extension
    from collections import Counter
    ext_counts = Counter(f["extension"] for f in results)
    print("\nFiles by extension:")
    for ext, count in ext_counts.most_common():
        label = ext if ext else "(no ext)"
        print(f"  {label:12s} → {count}")

    # Show all files
    print(f"\nAll files:")
    for f in sorted(results, key=lambda x: x["relative_path"]):
        print(f"  {f['relative_path']:60s} ({f['size_kb']:6.1f} KB)")

