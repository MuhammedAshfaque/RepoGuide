# CodeCompass — Project Progress

> **Tagline:** Ask your codebase anything. Get answers with file and line citations.

---

## Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| **1** | Setup and ingestion | ✅ Complete |
| **2** | Code-aware chunking | ✅ Complete |
| **3** | Indexing and basic retrieval | ✅ Complete |
| **4** | Hybrid search (BM25 + RRF) | ✅ Complete |
| **5** | Generation with citations | ⏳ Up next |
| **6** | Streamlit UI | 🔲 Not started |
| **7** | Evaluation | 🔲 Not started |
| **8** | Polish and deploy | 🔲 Not started |

---

## Current Phase: 5 — Generation with citations

**Goal:** Ollama LLM wrapper, strict prompt ("answer only from the provided code; cite as `path:start-end`; say you don't know if context is insufficient"), few-shot examples, post-verification check for citations, and clear error handling.

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
│   │   ├── vector_search.py      # Queries ChromaDB for semantically similar chunks.
│   │   │                           Embeds query, finds nearest neighbors, returns scored results.
│   │   ├── keyword_search.py     # BM25Okapi search with identifier-aware tokenization
│   │   │                           (snake_case & camelCase splitting). Saves/loads .pkl indexes.
│   │   └── hybrid.py             # Reciprocal Rank Fusion (RRF) combining vector and BM25 search.
│   │                               Includes CLI comparison tool (Vector vs BM25 vs Hybrid).
│   │
│   └── generation/
│       └── __init__.py           # (Phase 5: LLM wrapper, prompts, answerer)
│
├── tests/
│   ├── __init__.py
│   ├── test_loader.py            # 25 tests: URL parsing, filtering, integration
│   ├── test_chunker.py           # 49 tests: AST chunker, fallback chunker, edge cases
│   └── test_hybrid.py            # 19 tests: tokenization, BM25 indexing, RRF ranking math
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
| `tests/test_hybrid.py` | **19 pytest tests** across 3 test classes: subword identifier splitting (`snake_case`, `camelCase`, etc.), code-aware tokenization, BM25 indexing/search matching, and RRF rank fusion formula verification. Total test suite: **93 passed**. |
| `src/ingestion/chunker.py` | **Code-aware chunker.** AST-based extraction for `.py` files (one chunk per function/class/method + a module-level chunk). Falls back to line-based chunking (50 lines, 10 overlap) for non-Python files. Each chunk is a `Chunk` dataclass with `content`, `file_path`, `start_line`, `end_line`, `symbol_name`, `chunk_type`, `language`, `parent_class`, `decorators`, `has_docstring`. |
| `src/ingestion/indexer.py` | **Full indexing pipeline.** Orchestrates load → chunk → embed → store. Embeds in batches of 32 to respect 8 GB RAM. Uses ChromaDB `PersistentClient` so indexes survive between runs. Builds and persists both ChromaDB vector index and BM25 index. Supports `force_reindex`. CLI: `python -m src.ingestion.indexer <source>`. |
| `src/retrieval/vector_search.py` | **Semantic search.** Embeds the user's question with the same model, queries ChromaDB for top-k nearest neighbors using cosine similarity. Converts ChromaDB distances to 0–1 similarity scores. Returns results with `content`, `file_path`, `start_line`, `end_line`, `symbol_name`, `score`, `rank`. CLI: `python -m src.retrieval.vector_search <repo> <query>`. |
| `src/retrieval/keyword_search.py` | **BM25 keyword search.** Code-aware tokenization preserving original compound identifiers and constituent subwords. Implements `BM25Okapi` search and persistence (`.pkl`). Returns identically structured result dicts. CLI: `python -m src.retrieval.keyword_search <repo> <query>`. |
| `src/retrieval/hybrid.py` | **Hybrid retrieval via RRF.** Implements Reciprocal Rank Fusion ($k=60$) merging dense vector and sparse BM25 rankings. Provides `hybrid_search()` returning top candidates and a comparison CLI tool: `python -m src.retrieval.hybrid <repo> <query>`. |

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
| **Identifier-aware tokenization** | Code contains identifiers like `get_user_by_id` or `UserProfileView`. Standard tokenizers treat them as opaque words; code tokenization splits on `_` and camelCase while keeping the full identifier, matching both specific and subword queries. |
| **Reciprocal Rank Fusion (RRF)** | Dense vector cosine scores ($[0, 1]$) and BM25 scores ($[0, \infty)$) have incompatible scales. RRF uses ranks instead of raw scores ($1/(60 + \text{rank})$), eliminating fragile score normalization and hyperparameter tuning. |

---

## Smoke Test Results

**Demo repo:** `gothinkster/flask-realworld-example-app`

| Metric | Value |
|--------|-------|
| Total files loaded | 41 |
| Total chunks generated | 192 |
| Total test suite | **93 passed (100%)** |
| ChromaDB vector index | Persistent at `data/chroma_db/` |
| BM25 index | Serialized at `data/flask-realworld-example-app_bm25.pkl` |

**Retrieval Comparison for Query: `"jwt_optional"`**

| Search Mode | Top Result | Location | Score |
|-------------|------------|----------|-------|
| **Vector only** | `jwt_identity` | `conduit/utils.py:6-7` | 0.7190 |
| **BM25 only** | `profile` route module | `conduit/profile/views.py:1-35` | 9.8046 |
| **Hybrid (RRF)** | `profile` route module & `get_profile` | `conduit/profile/views.py:14-21` | RRF 0.0323 |

*(Notice how BM25 immediately located the exact `@jwt_optional` decorator routes, which RRF elevated to the top!)*

---

## Next Step

Phase 5: Build `src/generation/llm.py` (Ollama provider-agnostic wrapper), `src/generation/prompts.py` (strict citation prompt with few-shot example), and `src/generation/answerer.py` (post-verification check).

