# CodeCompass — Project Progress

> **Tagline:** Ask your codebase anything. Get answers with file and line citations.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Setup and ingestion | ✅ Complete |
| **2** | Code-aware chunking | ✅ Complete |
| **3** | Indexing and basic retrieval | ✅ Complete |
| **4** | Hybrid search (BM25 + RRF) | ⏳ Up next |
| **5** | Generation with citations | 🔲 Not started |
| **6** | Streamlit UI | 🔲 Not started |
| **7** | Evaluation | 🔲 Not started |
| **8** | Polish and deploy | 🔲 Not started |

---

## Current Phase: 4 — Hybrid search (BM25 + RRF)

**Goal:** Add BM25 keyword search with identifier-aware tokenization, Reciprocal Rank Fusion, and a CLI to compare vector-only, BM25-only, and hybrid results.

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
│   │   ├── loader.py             # Clones GitHub repos (shallow), walks directories,
│   │   │                           filters out junk files. Returns list of indexable files.
│   │   ├── chunker.py            # AST chunker for .py (functions, classes, methods)
│   │   │                           + fallback line-based chunker for other files.
│   │   └── indexer.py            # Full pipeline: chunk → embed (batch 32) → ChromaDB.
│   │                               Lazy model loading, persistent storage on disk.
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── vector_search.py      # Queries ChromaDB for semantically similar chunks.
│   │                               Embeds query, finds nearest neighbors, returns scored results.
│   │
│   └── generation/
│       └── __init__.py           # (Phase 5: LLM wrapper, prompts, answerer)
│
├── tests/
│   ├── __init__.py
│   └── test_chunker.py           # 49 tests: function extraction, classes, methods,
│                                   async, decorators, docstrings, nested funcs,
│                                   empty files, syntax errors, module-level code,
│                                   fallback chunking, language detection, routing
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
| `tests/test_chunker.py` | **49 pytest tests** across 13 test classes: basic functions, multiple functions, class with/without methods, async functions & methods, decorators, docstrings, nested functions, empty files, syntax errors (fallback), module-level code, `to_dict()` serialization, fallback line chunking, language detection, and `chunk_file()` routing. |
| `src/ingestion/chunker.py` | **Code-aware chunker.** AST-based extraction for `.py` files (one chunk per function/class/method + a module-level chunk). Falls back to line-based chunking (50 lines, 10 overlap) for non-Python files. Each chunk is a `Chunk` dataclass with `content`, `file_path`, `start_line`, `end_line`, `symbol_name`, `chunk_type`, `language`, `parent_class`, `decorators`, `has_docstring`. |
| `src/ingestion/indexer.py` | **Full indexing pipeline.** Orchestrates load → chunk → embed → store. Embeds in batches of 32 to respect 8 GB RAM. Uses ChromaDB `PersistentClient` so indexes survive between runs. Lazy-loads the embedding model (only loaded once). Supports `force_reindex` to rebuild and skips if already indexed. Also runnable as CLI: `python -m src.ingestion.indexer <source>`. |
| `src/retrieval/vector_search.py` | **Semantic search.** Embeds the user's question with the same model, queries ChromaDB for top-k nearest neighbors using cosine similarity. Converts ChromaDB distances to 0–1 similarity scores. Returns results with `content`, `file_path`, `start_line`, `end_line`, `symbol_name`, `score`, `rank`. CLI: `python -m src.retrieval.vector_search <repo> <query>`. |

---

## Design Decisions Log

| Decision | Rationale |
|----------|-----------|
| **Allowlist for file extensions** | Safer than a blocklist — unknown file types (`.wasm`, `.dat`) are rejected by default instead of accidentally embedded. |
| **Shallow clone (`depth=1`)** | We only need the latest code, not Git history. Saves disk and time on an 8 GB RAM machine. |
| **100 KB file size limit** | Files larger than this are usually generated code, data files, or minified bundles — waste of embedding memory and retrieval noise. |

| **One chunk = one function/class** | Fixed-size chunks split code at arbitrary boundaries, so a search for `get_user_by_id` might return half the function plus unrelated code. AST chunking guarantees every chunk is a complete semantic unit. |
| **50-line fallback with 10-line overlap** | For non-Python files we can't parse an AST, so we use fixed windows. The 10-line overlap prevents losing context at chunk boundaries. |
| **Module-level chunk** | Top-level code (imports, constants, globals) is also valuable for understanding a file. We extract it as a separate "module" chunk so retrieval can find things like `DATABASE_URI = ...`. |

| **Batch embedding (size 32)** | On an 8 GB RAM machine, embedding all 192 chunks at once would spike memory. Batching keeps peak RAM predictable — only 32 texts + their embeddings are in memory at any time. |
| **Cosine similarity in ChromaDB** | Cosine measures angle between vectors, ignoring magnitude. Two texts about the same topic point in similar directions even if one is much longer. This is the standard for text embedding search. |
| **Persistent ChromaDB** | Using `PersistentClient` stores the index on disk in `data/chroma_db/`. Re-running the app reuses the existing index without re-embedding — saves time and memory. |
| **Prefix embedding with symbol name** | Before embedding, we prepend the chunk type and symbol name (e.g. `"function login_user: @blueprint.route..."`) so the embedding captures *what the code is*, not just the raw text. |

---

## Smoke Test Results

**Demo repo:** `gothinkster/flask-realworld-example-app`

| Metric | Value |
|--------|-------|
| Total files loaded | 41 |
| Total chunks generated | 192 |
| Indexing time | 92.92s (includes model download) |
| Embedding model | all-MiniLM-L6-v2 (~90 MB) |

**Sample search: "Where is authentication handled?"**

| Rank | Score | Symbol | File |
|------|-------|--------|------|
| 1 | 0.69 | class `TestAuthenticate` | tests/test_authentication.py:17 |
| 2 | 0.66 | function `login_user` | conduit/user/views.py:30-40 |
| 3 | 0.66 | function `get_user` | conduit/user/views.py:43-50 |
| 4 | 0.63 | method `test_check_password` | tests/test_models.py:49-54 |
| 5 | 0.63 | function `register_user` | conduit/user/views.py:17-27 |

---

## Next Step

Phase 4: Build `src/retrieval/keyword_search.py` (BM25) and `src/retrieval/hybrid.py` (RRF fusion), plus a comparison CLI.

