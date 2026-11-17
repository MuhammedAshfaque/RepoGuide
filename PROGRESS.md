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
| **8** | Polish and deploy | ✅ Complete |

---

## Project Status: All 8 Phases Complete 🎉

**Goal:** Build a working, memory-efficient RAG codebase assistant on Windows (8 GB RAM) capable of answering architecture and code questions with verified line-level citations, accompanied by an empirical benchmark and documentation for AI engineering interviews.

**Final Summary:**
- **Full Test Suite:** **138 tests passed** (100% pass rate in ~5.5 seconds).
- **Retrieval Recall (Hit@5):** **96.9%** across 32 benchmark questions on `flask-realworld-example-app`.
- **Ranking Precision (MRR):** **0.8177** with Cross-Encoder reranker enabled (Hit@1 of **71.9%**).
- **Frontend:** Streamlit web interface with real-time streaming, verified citation badges, code snippet viewer, and live search mode switcher.
- **Documentation:** Complete `README.md` and a 3-minute interview demo script (`DEMO_PLAN.md`).

---

## File Map

```
codecompass/
│
├── .env                          # Local settings (gitignored)
├── .env.example                  # Template showing all config keys
├── .gitignore                    # Ignores venv/, data/, .env, etc.
├── requirements.txt              # Production dependencies
├── PROGRESS.md                   # ← This file — project progress tracker
├── README.md                     # Authoritative project documentation & architecture guide
├── DEMO_PLAN.md                  # 3-minute interview presentation script & walkthrough
├── app.py                        # Streamlit web UI (streaming chat, snippets, mode toggle)
│
├── src/
│   ├── __init__.py               # Makes src a Python package
│   ├── config.py                 # Central config: loads .env, defines filtering rules & models
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py             # Clones GitHub repos (shallow), walks directories, filters files
│   │   ├── chunker.py            # AST chunker for .py + fallback line-based chunker
│   │   └── indexer.py            # Batch embeddings, ChromaDB + BM25, content fingerprint caching
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_search.py      # Cosine similarity vector search over ChromaDB
│   │   ├── keyword_search.py     # BM25Okapi search with snake_case/camelCase identifier tokenization
│   │   ├── hybrid.py             # Reciprocal Rank Fusion (RRF, k=60) combining vector + BM25
│   │   └── reranker.py           # Cross-Encoder (ms-marco-MiniLM-L-6-v2) candidate re-scoring
│   │
│   └── generation/
│       ├── __init__.py
│       ├── llm.py                # Swappable LLM interface: OllamaProvider + GeminiProvider (REST)
│       ├── prompts.py            # Strict grounding prompt with few-shot citation examples
│       └── answerer.py           # End-to-end RAG pipeline + citation regex post-verifier
│
├── tests/
│   ├── __init__.py
│   ├── test_loader.py            # 25 tests: URL parsing, directory skip logic, file inclusion rules
│   ├── test_chunker.py           # 49 tests: AST chunker, fallback chunker, decorators, docstrings
│   ├── test_hybrid.py            # 19 tests: tokenization, BM25 indexing, RRF ranking math
│   ├── test_generation.py        # 13 tests: prompt rules, citation extraction & verification, mock LLM
│   ├── test_ui.py                # 5 tests: repo name resolver, index check, retrieval routing
│   ├── test_evaluation.py        # 15 tests: ground truth matching, Hit@k & MRR math, report generator
│   └── test_reranker.py          # 12 tests: cross-encoder reranking, index caching, Gemini provider
│
├── evaluation/
│   ├── eval_questions.json       # 32 ground-truth benchmark questions across 7 categories
│   ├── run_eval.py               # Benchmark runner: evaluates Vector, BM25, Hybrid, and Rerank
│   └── results.md                # Quantitative evaluation report with comparison tables
│
└── data/                         # Cloned repos + indexes (gitignored)
    └── flask-realworld-example-app/  # Demo repo (41 files, 192 chunks)
```

---

### What each file does

| File | Role |
|------|------|
| `src/config.py` | **Single source of truth for settings.** Loads `.env`, exposes constants like `LLM_MODEL`, `MAX_FILE_SIZE_KB`, `RERANKER_MODEL`, `GEMINI_API_KEY`, and filtering sets (`SKIP_DIRS`, `ALLOWED_EXTENSIONS`, `BINARY_EXTENSIONS`, `SKIP_FILES`). |
| `src/ingestion/loader.py` | **Repo loader.** (1) Clones a GitHub URL via `git clone --depth 1`, (2) walks the directory tree, (3) filters out `.git`, `node_modules`, binaries, lock files, oversized files. Returns list of file dicts. |
| `src/ingestion/chunker.py` | **Code-aware chunker.** AST-based extraction for `.py` files (one chunk per function/class/method + module chunk). Falls back to line-based chunking (50 lines, 10 overlap) for non-Python files. Preserves decorators, parent class, and docstrings. |
| `src/ingestion/indexer.py` | **Full indexing pipeline with caching.** Computes content fingerprint (Git SHA or MD5 hash of mtimes and file paths). If repo is unchanged, re-indexing is skipped in <3 seconds. Otherwise embeds chunks in batches of 32, populates ChromaDB, and builds BM25 index. |
| `src/retrieval/vector_search.py` | **Semantic search.** Embeds query using `all-MiniLM-L6-v2`, queries ChromaDB for top-k nearest neighbors using cosine similarity. Converts ChromaDB distances to 0–1 similarity scores. |
| `src/retrieval/keyword_search.py` | **BM25 keyword search.** Code-aware tokenization splitting on `_` and camelCase while preserving original compound tokens. Implements `BM25Okapi` search and persistence (`.pkl`). |
| `src/retrieval/hybrid.py` | **Hybrid retrieval via RRF.** Implements Reciprocal Rank Fusion ($k=60$) merging dense vector and sparse BM25 rankings. Provides CLI comparison tool: `python -m src.retrieval.hybrid <repo> <query>`. |
| `src/retrieval/reranker.py` | **Cross-Encoder reranker.** Lazy-loads `ms-marco-MiniLM-L-6-v2` (~80 MB) to re-score top-20 hybrid candidates via full cross-attention. Includes automatic fallback if model is unvailable. |
| `src/generation/llm.py` | **Provider-agnostic LLM interface.** Abstract `LLMProvider` with `OllamaProvider` (local `qwen2.5-coder:3b`, `num_ctx=4096`, `temperature=0.1`) and `GeminiProvider` (zero-dependency REST API for cloud hosting). |
| `src/generation/prompts.py` | **Strict prompt templates.** Instructs LLM to answer only from context snippets, cite exact lines (`path:start-end`), and return `"I couldn't find this in the codebase"` if context is insufficient. Includes few-shot positive and negative citation examples. |
| `src/generation/answerer.py` | **End-to-end RAG question answering.** Coordinates retrieval → prompt formatting → LLM generation → citation post-verification. Validates that cited file paths and line numbers genuinely overlap with retrieved chunks. |
| `app.py` | **Interactive Streamlit web application.** Provides repo URL / path input, one-click indexing with progress spinner, real-time token streaming chat interface, post-generation citation validation display, expandable syntax-highlighted code blocks, and a live toggle between Hybrid, Hybrid+Rerank, Vector-only, and BM25-only search modes. |
| `evaluation/eval_questions.json` | **Benchmark ground-truth dataset.** 32 realistic developer questions covering architecture, authentication, database models, profiles, articles, errors, and configuration. |
| `evaluation/run_eval.py` | **Retrieval evaluation engine.** Evaluates questions across Vector, BM25, Hybrid, and Hybrid+Rerank modes. Calculates `first_rank`, reciprocal rank, Hit@1, Hit@3, Hit@5, MRR, and latency per mode. |
| `evaluation/results.md` | **Formal benchmark report.** Detailed markdown document with full comparison tables and architectural analysis. |
| `DEMO_PLAN.md` | **Interview presentation guide.** Step-by-step 3-minute walkthrough script with sample questions, talking points, and memory metrics for interview demos. |
| `README.md` | **Production portfolio documentation.** Complete project overview, Mermaid architecture diagram, evaluation table, design rationales, limitations, and Windows setup instructions. |

---

## Complete Test Suite Breakdown (138 Tests)

| Test File | Tests | Focus Areas |
|---|:---:|---|
| `tests/test_loader.py` | 25 | GitHub URL parsing, directory ignore rules, file extension allowlists, size limits, binary rejection |
| `tests/test_chunker.py` | 49 | AST extraction (functions, classes, methods, async, decorators, docstrings, nested), fallback chunker, syntax errors |
| `tests/test_hybrid.py` | 19 | Identifier splitting (`snake_case`, `camelCase`), BM25 persistence, RRF fusion math |
| `tests/test_generation.py` | 13 | Prompt grounding rules, few-shot examples, citation regex parsing, verification audit, mock LLM pipeline |
| `tests/test_ui.py` | 5 | Repo name resolution, index check helpers, search mode engine routing |
| `tests/test_evaluation.py` | 15 | Chunk relevance matching, symbol hierarchy, single-query ranking evaluation, aggregate metric math, report generator |
| `tests/test_reranker.py` | 12 | Candidate re-scoring, rank reassignment, graceful fallback, repo fingerprinting, Gemini provider payload & health checks |
| **Total** | **138** | **100% Passing in ~5.5 seconds** |

---

## Benchmark Evaluation Results

**Target Repository:** `gothinkster/flask-realworld-example-app` (32 ground-truth questions)

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|:---:|:---:|:---:|:---:|
| **Hit@1** | 62.5% | 40.6% | **43.8%** | **71.9%** |
| **Hit@3** | 87.5% | 62.5% | **87.5%** | **90.6%** |
| **Hit@5** | 90.6% | 68.8% | **96.9%** | **96.9%** |
| **MRR (Mean Reciprocal Rank)** | 0.7474 | 0.5104 | **0.6557** | **0.8177** |
| **Average Latency (ms)** | 37.3 ms | 1.5 ms | 39.1 ms | 2587.5 ms |

### Category Breakdown

| Category | Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|---|:---:|:---:|:---:|:---:|
| **Conceptual / Natural Language** *(22 queries)* | **Hit@5** | 86.4% | 54.5% | **95.5%** | 95.5% |
| | **MRR** | 0.7008 | 0.3333 | **0.5674** | **0.8333** |
| **Exact Symbol / Identifier** *(10 queries)* | **Hit@5** | 100.0% | 100.0% | **100.0%** | 100.0% |
| | **MRR** | 0.8500 | **0.9000** | 0.8500 | 0.7833 |

---

## Design Decisions Log

| Decision | Rationale |
|----------|-----------|
| **Allowlist for file extensions** | Safer than a blocklist — unknown file types (`.wasm`, `.dat`) are rejected by default instead of accidentally embedded. |
| **Shallow clone (`depth=1`)** | We only need the latest code, not Git history. Saves disk and time on an 8 GB RAM machine. |
| **100 KB file size limit** | Files larger than this are usually generated code, data files, or minified bundles — waste of embedding memory and retrieval noise. |
| **One chunk = one function/class** | Fixed-size chunks split code at arbitrary boundaries. AST chunking guarantees every chunk is a complete semantic unit. |
| **50-line fallback with 10-line overlap** | For non-Python files we can't parse an AST, so we use fixed windows. The 10-line overlap prevents losing context at chunk boundaries. |
| **Module-level chunk** | Top-level code (imports, constants, globals) is extracted as a separate "module" chunk so retrieval can find things like `DATABASE_URI = ...`. |
| **Batch embedding (size 32)** | On an 8 GB RAM machine, embedding all 192 chunks at once would spike memory. Batching keeps peak RAM predictable. |
| **Cosine similarity in ChromaDB** | Cosine measures angle between vectors, ignoring magnitude. The standard for semantic embedding search. |
| **Persistent ChromaDB** | Using `PersistentClient` stores the index on disk in `data/chroma_db/`. Re-running the app reuses the existing index. |
| **Prefix embedding with symbol name** | Before embedding, we prepend chunk type and symbol name (e.g. `"function login_user: @blueprint.route..."`) so the embedding captures what the code is. |
| **Identifier-aware tokenization** | Code contains compound identifiers like `get_user_by_id`. Subword splitting on `_` and camelCase allows BM25 to match both exact symbols and partial keywords. |
| **Reciprocal Rank Fusion (RRF)** | Dense cosine scores ($[0, 1]$) and BM25 scores ($[0, \infty)$) have incompatible scales. RRF uses rank positions ($1/(60 + \text{rank})$), eliminating score calibration. |
| **Few-shot citations for 3B LLM** | Small models like `qwen2.5-coder:3b` follow format rules less reliably. Few-shot examples anchor exact citation syntax (`file.py:start-end`) and grounded negative responses. |
| **Citation post-verification** | Programmatically regex-extracts every citation in generated text and checks that the cited file and lines actually overlap with retrieved chunks, detecting hallucinations. |
| **Provider-agnostic LLM interface** | All generation logic is decoupled behind the `LLMProvider` abstract base class, allowing effortless switching to Google Gemini without modifying UI or answerer. |
| **Token streaming in chat UI** | Local 3B models on CPU/consumer RAM take 5–15s for full multi-paragraph responses. Token streaming drops perceived latency to <1s. |
| **Repo Content Fingerprinting** | Computes Git SHA or MD5 hash over file paths, sizes, and mtimes. Unchanged repositories skip re-indexing immediately (<3 seconds). |
| **Cross-Encoder Reranker Trade-off** | `ms-marco-MiniLM-L-6-v2` increases Hit@1 to 71.9% and MRR to 0.8177, but increases CPU latency from 42 ms to 3.0 s. Made optional via UI and CLI. |
