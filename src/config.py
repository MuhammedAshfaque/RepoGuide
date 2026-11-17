"""
Configuration module for CodeCompass.

Loads settings from a .env file and provides them as typed constants.
All configurable values live here so nothing is hardcoded elsewhere.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env from project root ──────────────────────────────────────────────
# The project root is the parent of the `src/` folder this file lives in.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"           # cloned repos + indexes stored here
DATA_DIR.mkdir(exist_ok=True)

# ── LLM Settings ─────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5-coder:3b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
LLM_NUM_CTX: int = 4096          # context window size for Ollama
LLM_TEMPERATURE: float = 0.1    # low temperature → deterministic answers

# ── Embedding Settings ───────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Retrieval Settings ───────────────────────────────────────────────────────
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", "4"))

# ── Indexing Settings ────────────────────────────────────────────────────────
MAX_FILE_SIZE_KB: int = int(os.getenv("MAX_FILE_SIZE_KB", "100"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ── File Filtering ───────────────────────────────────────────────────────────
# Directories to always skip during ingestion.
SKIP_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", "venv", ".venv",
    "env", ".env", ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build", "egg-info", ".eggs", ".idea", ".vscode",
}

# File extensions we consider as source code / documentation.
ALLOWED_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".rb", ".php",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".md", ".txt", ".rst", ".yaml", ".yml",
    ".json", ".toml", ".cfg", ".ini",
    ".html", ".css", ".scss",
    ".sh", ".bat", ".ps1",
    ".sql",
    ".dockerfile",
}

# Binary / media extensions to always reject (even if under size limit).
BINARY_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".exe", ".dll", ".so", ".dylib", ".o", ".obj",
    ".pyc", ".pyo", ".class", ".wasm",
    ".ttf", ".woff", ".woff2", ".eot",
    ".sqlite", ".db",
}

# Files whose exact name should be skipped.
SKIP_FILES: set[str] = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "composer.lock",
    ".DS_Store", "Thumbs.db",
}


def get_collection_name(repo_name: str) -> str:
    """
    Build a valid collection / index identifier from a repo name.

    ChromaDB collection names must:
      - Be 3-63 characters
      - Start and end with alphanumeric
      - Contain only alphanumeric, underscores, or hyphens
    """
    clean = "".join(c if c.isalnum() or c in "-_" else "-" for c in repo_name)
    clean = clean.strip("-_")
    if len(clean) < 3:
        clean = clean + "-idx"
    return clean[:63]


