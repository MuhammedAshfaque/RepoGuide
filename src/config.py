"""Central configuration loaded from .env."""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── LLM settings (local Ollama default; optional Gemini via .env) ────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen2.5:7b")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
LLM_NUM_CTX: int = 4096
LLM_TEMPERATURE: float = 0.1

# ── Embedding, reranker, and retrieval settings (sized for 8 GB RAM) ─────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", "4"))
MAX_FILE_SIZE_KB: int = int(os.getenv("MAX_FILE_SIZE_KB", "100"))
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# ── File filtering rules ─────────────────────────────────────────────────────
SKIP_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", "venv", ".venv",
    "env", ".env", ".tox", ".mypy_cache", ".pytest_cache",
    "dist", "build", "egg-info", ".eggs", ".idea", ".vscode",
}

ALLOWED_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".rb", ".php",
    ".c", ".cpp", ".h", ".hpp", ".cs",
    ".md", ".txt", ".rst", ".yaml", ".yml",
    ".json", ".toml", ".cfg", ".ini",
    ".html", ".css", ".scss",
    ".sh", ".bat", ".ps1", ".sql", ".dockerfile",
}

SKIP_FILES: set[str] = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "poetry.lock", "Pipfile.lock", "composer.lock",
    ".DS_Store", "Thumbs.db",
}
