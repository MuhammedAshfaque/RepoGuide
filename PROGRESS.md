# CodeCompass — Project Progress

> **Tagline:** Ask your codebase anything. Get answers with file and line citations.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Setup and ingestion | ✅ Complete |
| **2** | Code-aware chunking | ⏳ Up next |
| **3** | Indexing and basic retrieval | 🔲 Not started |
| **4** | Hybrid search (BM25 + RRF) | 🔲 Not started |
| **5** | Generation with citations | 🔲 Not started |
| **6** | Streamlit UI | 🔲 Not started |
| **7** | Evaluation | 🔲 Not started |
| **8** | Polish and deploy | 🔲 Not started |

---

## Current Phase: 2 — Code-aware chunking

**Goal:** Use Python's `ast` module to extract functions, classes, and methods with exact line numbers. Add fallback line-based chunking for non-Python files. Write pytest tests.

**Status:** Waiting for user to say "continue".

---

## File Map

```
codecompass/
│
├── .env                          # Local settings (gitignored)
├── .env.example                  # Template showing all config keys
├── .gitignore                    # Ignores venv/, data/, .env, etc.
├── requirements.txt              # Dependencies (Phase 1 active, rest commented)
├── PROGRESS.md                   # ← This file — project progress tracker
│
├── src/
│   ├── __init__.py               # Makes src a Python package
│   ├── config.py                 # Central config: loads .env, defines filtering
│   │                               rules (skip dirs, allowed extensions, size limits)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── loader.py             # Clones GitHub repos (shallow), walks directories,
│   │                               filters out junk files. Returns list of indexable files.
│   │
│   ├── retrieval/
│   │   └── __init__.py           # (Phase 3–4: vector search, BM25, hybrid fusion)
│   │
│   └── generation/
│       └── __init__.py           # (Phase 5: LLM wrapper, prompts, answerer)
│
├── tests/
│   ├── __init__.py
│   └── test_loader.py            # 25 tests: URL parsing, dir skipping, file
│                                   filtering (extensions, size, binaries), integration
│
├── evaluation/                   # (Phase 7: eval questions, metrics, results)
│
└── data/                         # Cloned repos + indexes (gitignored)
    └── flask-realworld-example-app/  # Demo repo (41 files, used for testing)
```

### What each file does

| File | Role |
|------|------|
| `src/config.py` | **Single source of truth for settings.** Loads `.env`, exposes constants like `LLM_MODEL`, `MAX_FILE_SIZE_KB`, and filtering sets (`SKIP_DIRS`, `ALLOWED_EXTENSIONS`, `BINARY_EXTENSIONS`, `SKIP_FILES`). Every other module imports from here. |
| `src/ingestion/loader.py` | **Repo loader.** (1) Clones a GitHub URL via `git clone --depth 1`, (2) walks the directory tree, (3) filters out `.git`, `node_modules`, binaries, lock files, oversized files. Returns a list of dicts with `path`, `relative_path`, `extension`, `size_kb`. Also runnable as a CLI: `python -m src.ingestion.loader <url>`. |
| `tests/test_loader.py` | **25 pytest tests** across 4 test classes: URL parsing, directory skip logic, file inclusion rules (extensions, size limits, binary rejection, no-extension files), and a full integration test with a fake repo directory. |

---

## Design Decisions Log

| Decision | Rationale |
|----------|-----------|
| **Allowlist for file extensions** | Safer than a blocklist — unknown file types (`.wasm`, `.dat`) are rejected by default instead of accidentally embedded. |
| **Shallow clone (`depth=1`)** | We only need the latest code, not Git history. Saves disk and time on an 8 GB RAM machine. |
| **100 KB file size limit** | Files larger than this are usually generated code, data files, or minified bundles — waste of embedding memory and retrieval noise. |

---

## Next Step

Phase 2: Build `src/ingestion/chunker.py` — AST-based chunking for `.py` files, fallback line-based chunking for everything else, with pytest tests.

