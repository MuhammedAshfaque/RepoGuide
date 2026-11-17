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
| **5** | Generation with citations | ✅ Complete |
| **6** | Streamlit UI | ✅ Complete |
| **7** | Evaluation | ✅ Complete |
| **8** | Polish and deploy | ⏳ Up next |

---

## Current Phase: 7 — Evaluation (Completed)

**Goal:** Create `evaluation/eval_questions.json` with 30+ ground-truth Q&A pairs (expected file/symbol), implement `evaluation/run_eval.py` computing Hit@k and MRR across Vector, BM25, and Hybrid modes, and output comparison table in `evaluation/results.md`.

**Status:** Completed. 15 evaluation unit tests passed (126 total tests passing). Benchmark executed across 32 queries with `results.md` generated. Waiting for user to say "continue" before starting Phase 8.

---

## File Map

```
codecompass/
│
├── .env                          # Local settings (gitignored)
├── .env.example                  # Template showing all config keys
├── .gitignore                    # Ignores venv/, data/, .env, etc.
├── requirements.txt              # Dependencies (Phases 1-7 active)
├── PROGRESS.md                   # ← This file — project progress tracker
├── app.py                        # Streamlit web UI (streaming chat, snippets, mode toggle)
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
│       ├── __init__.py
│       ├── llm.py                # Provider-agnostic LLM interface + OllamaProvider (num_ctx=4096,
│       │                           temperature=0.1, error diagnostics for missing models/daemon).
│       ├── prompts.py            # Strict grounding system prompt + few-shot citation examples
│       │                           + chunk context formatter with truncation safeguards.
│       └── answerer.py           # End-to-end RAG pipeline (retrieve -> prompt -> generate ->
│                                   verify citations). Verifies file paths and line ranges.
│
├── tests/
│   ├── __init__.py
│   ├── test_loader.py            # 25 tests: URL parsing, filtering, integration
│   ├── test_chunker.py           # 49 tests: AST chunker, fallback chunker, edge cases
│   ├── test_hybrid.py            # 19 tests: tokenization, BM25 indexing, RRF ranking math
│   ├── test_generation.py        # 13 tests: prompt rules, citation regex & verification, mock LLM
│   ├── test_ui.py                # 5 tests: repo name resolver, index check, retrieval routing
│   └── test_evaluation.py        # 15 tests: ground truth matching, Hit@k & MRR math, markdown report
│
├── evaluation/
│   ├── eval_questions.json       # 32 ground-truth benchmark questions across 7 categories
│   ├── run_eval.py               # Benchmark runner: evaluates Vector, BM25, and Hybrid
│   └── results.md                # Generated comparative evaluation report with tables and analysis
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
| `tests/test_hybrid.py` | **19 pytest tests** across 3 test classes: subword identifier splitting (`snake_case`, `camelCase`, etc.), code-aware tokenization, BM25 indexing/search matching, and RRF rank fusion formula verification. |
| `src/ingestion/chunker.py` | **Code-aware chunker.** AST-based extraction for `.py` files (one chunk per function/class/method + a module-level chunk). Falls back to line-based chunking (50 lines, 10 overlap) for non-Python files. Each chunk is a `Chunk` dataclass with `content`, `file_path`, `start_line`, `end_line`, `symbol_name`, `chunk_type`, `language`, `parent_class`, `decorators`, `has_docstring`. |
| `src/ingestion/indexer.py` | **Full indexing pipeline.** Orchestrates load → chunk → embed → store. Embeds in batches of 32 to respect 8 GB RAM. Uses ChromaDB `PersistentClient` so indexes survive between runs. Builds and persists both ChromaDB vector index and BM25 index. Supports `force_reindex`. CLI: `python -m src.ingestion.indexer <source>`. |
| `src/retrieval/vector_search.py` | **Semantic search.** Embeds the user's question with the same model, queries ChromaDB for top-k nearest neighbors using cosine similarity. Converts ChromaDB distances to 0–1 similarity scores. Returns results with `content`, `file_path`, `start_line`, `end_line`, `symbol_name`, `score`, `rank`. CLI: `python -m src.retrieval.vector_search <repo> <query>`. |
| `src/retrieval/keyword_search.py` | **BM25 keyword search.** Code-aware tokenization preserving original compound identifiers and constituent subwords. Implements `BM25Okapi` search and persistence (`.pkl`). Returns identically structured result dicts. CLI: `python -m src.retrieval.keyword_search <repo> <query>`. |
| `src/retrieval/hybrid.py` | **Hybrid retrieval via RRF.** Implements Reciprocal Rank Fusion ($k=60$) merging dense vector and sparse BM25 rankings. Provides `hybrid_search()` returning top candidates and a comparison CLI tool: `python -m src.retrieval.hybrid <repo> <query>`. |
| `src/generation/llm.py` | **Provider-agnostic LLM interface.** Abstract `LLMProvider` with `generate()` and `stream()` methods. Concrete `OllamaProvider` connects to local Ollama (`qwen2.5-coder:3b`) with health checks, error diagnosis, `num_ctx=4096`, and `temperature=0.1`. CLI: `python -m src.generation.llm`. |
| `src/generation/prompts.py` | **Strict prompt templates.** Instructs LLM to answer only from context snippets, cite exact lines (`path:start-end`), and return `"I couldn't find this in the codebase"` if context is insufficient. Includes few-shot positive and negative citation examples. Formats chunks with truncation protection. |
| `src/generation/answerer.py` | **End-to-end RAG question answering.** Coordinates retrieval → prompt formatting → LLM generation → citation post-verification. Parses all citations in the answer and validates that the cited file and lines actually overlap with retrieved chunks. CLI: `python -m src.generation.answerer <repo> <query>`. |
| `tests/test_generation.py` | **13 pytest tests** covering prompt rules, few-shot presence, chunk truncation, citation extraction regex, citation verification logic (valid, boundary margin, out-of-range, wrong file), and full pipeline testing using a `MockLLMProvider`. |
| `tests/test_ui.py` | **5 pytest tests** covering UI helper functions: repository name resolution from URLs and local relative/absolute paths, index status checking, and retrieval engine routing across all three modes. |
| `app.py` | **Interactive Streamlit web application.** Provides repo URL / path input, one-click indexing with progress spinner, real-time token streaming chat interface, post-generation citation validation display, expandable syntax-highlighted code blocks, and a live toggle between Hybrid, Vector-only, and BM25-only search modes. |
| `evaluation/eval_questions.json` | **Benchmark ground-truth dataset.** 32 realistic developer questions covering architecture, authentication, database models, profiles, articles, errors, and configuration. Each entry specifies `question`, `expected_file`, `expected_symbol`, and `category`. |
| `evaluation/run_eval.py` | **Retrieval evaluation engine.** Evaluates questions across Vector, BM25, and Hybrid modes. Calculates `first_rank`, reciprocal rank, Hit@1, Hit@3, Hit@5, MRR, and latency per mode. Computes category breakdowns and outputs markdown reports. |
| `tests/test_evaluation.py` | **15 pytest tests** covering chunk relevance matching (exact, case-insensitive, backslash normalization, enclosing class, sub-symbol), single-query ranking evaluation, aggregate metric calculations, and report file generation. Total test suite: **126 passed**. |
| `evaluation/results.md` | **Formal benchmark report.** Compares Vector-only, BM25-only, and Hybrid retrieval across Hit@1, Hit@3, Hit@5, MRR, and query latency with architectural analysis of why hybrid search delivers a 96.9% Hit@5. |

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
| **Few-shot citations for 3B LLM** | Small models like `qwen2.5-coder:3b` are fast and memory-friendly on 8 GB RAM, but follow zero-shot format instructions unreliably. Providing concrete few-shot examples anchors exact citation syntax (`file.py:start-end`) and grounded negative responses. |
| **Citation post-verification** | Programmatically regex-extracts every citation in the generated text and checks that the cited file path and line numbers genuinely overlap with retrieved chunks. Detects hallucinations before displaying answers to users. |
| **Provider-agnostic LLM interface** | All generation logic is decoupled behind the `LLMProvider` abstract base class, allowing effortless switching to cloud providers (Gemini, Groq, OpenAI) later without modifying the answerer or UI. |
| **Token streaming in chat UI** | Local 3B models on CPU/consumer RAM take 5–15s for full multi-paragraph responses. Token streaming drops perceived latency to <1s, providing immediate visual feedback as words appear. |
| **Search mode toggle in UI** | Allows users and interviewers to compare Vector, BM25, and Hybrid results dynamically on the same query to observe where semantic search vs exact keyword matching excels. |
| **Concurrency & RAM protection** | Prevents simultaneous indexing and LLM generation within the session state, avoiding memory spikes on 8 GB RAM systems. |
| **Hit@k and MRR metric selection** | Standard Information Retrieval (IR) metrics: Hit@5 directly assesses whether the right code chunk enters the LLM's top-4/5 prompt context window, while MRR measures how close to the #1 spot the correct chunk lands. |
| **Granular symbol relevance hierarchy** | Ground-truth relevance accounts for code hierarchy: a chunk is relevant if it matches the file and the exact symbol, a sub-symbol, an enclosing class (e.g. `User.check_password` matches when `User` is queried), or a module-level chunk for file-level architectural queries. |

---

## Benchmark Evaluation Results

**Evaluated on:** `gothinkster/flask-realworld-example-app` (32 queries across 7 functional categories)

| Metric | Vector Only | BM25 Only | Hybrid (RRF) |
|---|:---:|:---:|:---:|
| **Hit@1** | 62.5% | 40.6% | **43.8%** |
| **Hit@3** | 87.5% | 62.5% | **87.5%** |
| **Hit@5** | 90.6% | 68.8% | **96.9%** |
| **MRR (Mean Reciprocal Rank)** | 0.7474 | 0.5104 | **0.6557** |
| **Average Latency (ms)** | 1098.9 ms | 1.3 ms | 26.1 ms |

> **Key takeaway:** Hybrid search achieves a **96.9% Hit@5**, outperforming Vector-only (90.6%) and BM25-only (68.8%). When vector search misses configuration files or exact classes (e.g., `DevConfig`, `TestConfig`), BM25 surfaces them, and RRF pulls them into the top ranks.

---

## Smoke Test Results

**Demo repo:** `gothinkster/flask-realworld-example-app`

| Metric | Value |
|--------|-------|
| Total files loaded | 41 |
| Total chunks generated | 192 |
| Total test suite | **126 passed (100%)** |
| ChromaDB vector index | Persistent at `data/chroma_db/` |
| BM25 index | Serialized at `data/flask-realworld-example-app_bm25.pkl` |
| Web Interface | Streamlit on `http://localhost:8501` |
| Features verified | Repo indexing, streaming generation, citations audit, syntax-highlighted snippets, mode toggle, quantitative IR evaluation |

---

## Next Step

Phase 8: Polish and deploy (caching, graceful error handling for missing models and index corruption, comprehensive demo plan, and production-ready `README.md`).
