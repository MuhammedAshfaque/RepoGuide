# CodeCompass — Project Progress

> **Tagline:** Ask your codebase anything. Get answers with file and line citations.

---

## Project Status: LangChain Migration Complete 🎉

All 8 original phases (saved at git tag `manual-v1`) have been migrated cleanly to **LangChain** directly on `main`, removing redundant custom plumbing while preserving all features, custom code-specific components, and retrieval accuracy.

### Migration Steps Summary

| Step | Description | Status |
|------|-------------|--------|
| **Baseline** | All 8 manual phases complete (138 tests, git tag `manual-v1`, saved `evaluation/results_manual_baseline.md`) | ✅ Complete |
| **Step 1** | Convert AST & fallback chunker output (`Chunk`) to LangChain `Document` (`a0d54de`) | ✅ Complete |
| **Step 2** | Migrate indexer to `HuggingFaceEmbeddings`, `langchain_chroma.Chroma`, `docs.pkl`, keeping fingerprint cache (`7669caa`) | ✅ Complete |
| **Step 3** | Consolidate 4 manual retrieval modules into `src/retrieval/retrieval.py` (`BM25Retriever`, `EnsembleRetriever`, `CrossEncoderReranker`) (`f66959a`) | ✅ Complete |
| **Step 4** | Replace 3 manual generation modules with unified LCEL `src/generation/chain.py` (`ChatOllama`/`ChatGoogleGenerativeAI`, `ChatPromptTemplate`, `StrOutputParser`, citation verification) (`ac3a1d5`) | ✅ Complete |
| **Step 5** | Migrate `app.py` and `tests/test_ui.py` to LangChain retrievers and LCEL streaming (`006e175`) | ✅ Complete |
| **Step 6** | Adapt `evaluation/run_eval.py` to LangChain retrievers and benchmark (`86d0a24`) | ✅ Complete |
| **Change 1** | Restore metadata header (`# type parent.symbol \| File: path`) in `Document.page_content`, strip in UI/prompts, force re-index (`0154427`) | ✅ Complete |
| **Change 2** | Filter zero-score BM25 docs via `_PositiveBM25Retriever` (`4f20c0e`) | ✅ Complete |
| **Step 7** | Update `requirements.txt`, `README.md`, `PROGRESS.md`, and final LOC report | ✅ Complete |

---

## Lines-of-Code Comparison (`manual-v1` → LangChain)

| Category | `manual-v1` | LangChain | Net Change | Git Diff (Added / Removed) |
|---|:---:|:---:|:---:|:---:|
| **Source code (`src/` + `app.py`)** | 3,051 lines (16 files) | 2,226 lines (11 files) | **-825 lines (-27.0%)** | +1,136 / -1,961 |
| **Test code (`tests/`)** | 1,568 lines (8 files) | 1,424 lines (7 files) | **-144 lines (-9.2%)** | +486 / -630 |
| **Evaluation (`evaluation/run_eval.py`)** | 442 lines (1 file) | 438 lines (1 file) | **-4 lines (-0.9%)** | +71 / -75 |

**Deleted manual files:** `src/retrieval/vector_search.py`, `src/retrieval/keyword_search.py`, `src/retrieval/hybrid.py`, `src/retrieval/reranker.py`, `src/generation/llm.py`, `src/generation/prompts.py`, `src/generation/answerer.py`, `tests/test_hybrid.py`, `tests/test_reranker.py`, `tests/test_generation.py`.  
**Added LangChain files:** `src/retrieval/retrieval.py`, `src/generation/chain.py`, `tests/test_retrieval.py`, `tests/test_chain.py`.

---

## File Map

```text
codecompass/
├── .env                                 # Local settings (gitignored)
├── .env.example                         # Config template (Ollama default + optional Gemini key)
├── .gitignore                           # Ignores venv/, data/, .env
├── requirements.txt                     # LangChain + Chroma + BM25 + Streamlit dependencies
├── PROGRESS.md                          # ← This file
├── README.md                            # Documentation, Mermaid architecture & benchmark tables
├── DEMO_PLAN.md                         # 3-minute interview walkthrough script
├── app.py                               # Streamlit UI (indexing, streaming LCEL chat, snippets)
│
├── src/
│   ├── __init__.py
│   ├── config.py                        # Central .env config & file filtering rules
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                    # Shallow git clone, directory walk & file filtering (custom)
│   │   ├── chunker.py                   # Python AST chunker + 50-line fallback -> LangChain Documents (custom)
│   │   └── indexer.py                   # HuggingFaceEmbeddings, LangChain Chroma, header enrichment & fingerprint cache
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── retrieval.py                 # Vector, BM25Retriever (+ custom tokenizer), EnsembleRetriever, CrossEncoderReranker
│   └── generation/
│       ├── __init__.py
│       └── chain.py                     # ChatOllama/Gemini, ChatPromptTemplate, LCEL chain & custom citation verifier
│
├── tests/
│   ├── __init__.py
│   ├── test_loader.py                   # 25 tests — URL parsing, directory/file filters
│   ├── test_chunker.py                  # 50 tests — AST chunking, fallback chunker, LangChain Document
│   ├── test_retrieval.py                # 17 tests — identifier tokenizer, BM25, Ensemble RRF, Rerank & fallback
│   ├── test_chain.py                    # 15 tests — prompt rules, header enrichment, LCEL chain, citations
│   ├── test_ui.py                       # 5 tests  — repo name resolution, index check, mode routing
│   └── test_evaluation.py               # 16 tests — ground truth matching, Hit@k & MRR math, report generator
│
├── evaluation/
│   ├── eval_questions.json              # 32 ground-truth questions across 7 categories
│   ├── run_eval.py                      # Warm-up benchmark runner (Hit@k, MRR, latency)
│   ├── results.md                       # Current LangChain benchmark report
│   └── results_manual_baseline.md       # Saved pre-migration manual-v1 baseline report
│
└── data/                                # Cloned repos + indexes (gitignored)
    └── flask-realworld-example-app/     # Demo repo (41 files, 192 chunks)
```

---

## Complete Test Suite Breakdown (128 Tests)

| Test File | Tests | Focus Areas |
|---|:---:|---|
| `tests/test_loader.py` | 25 | GitHub URL parsing, directory skip rules, allowlist, binary rejection, size limits |
| `tests/test_chunker.py` | 50 | AST extraction (functions, classes, methods, async, decorators, docstrings, nested), fallback chunker, LangChain `Document` compliance |
| `tests/test_retrieval.py` | 17 | Identifier splitting (`snake_case`, `camelCase`), BM25 + zero-score filter, Ensemble RRF + slicing, CrossEncoder rerank + fallback, mode factory |
| `tests/test_chain.py` | 15 | `ChatPromptTemplate` rules & few-shot examples, metadata header enrichment/stripping, provider switching, LCEL invoke/stream, refusal, citation regex audit |
| `tests/test_ui.py` | 5 | Repo name resolver, index presence check, retrieval mode routing |
| `tests/test_evaluation.py` | 16 | Ground-truth matching (dicts & `Document`), Hit@k & MRR math, report generator |
| **Total** | **128** | **100% Passing** |

---

## Benchmark Results: Manual v1 vs. LangChain

**Target Repository:** `gothinkster/flask-realworld-example-app` (32 ground-truth questions, warm-up before timing)

| Mode | Version | Hit@1 | Hit@3 | Hit@5 | MRR | Avg Latency |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Vector Only** | Manual v1 | 62.5% | 87.5% | 90.6% | 0.7474 | 37.3 ms |
| **Vector Only** | LangChain | **65.6%** | **87.5%** | **90.6%** | **0.7526** | **24.8 ms** |
| **BM25 Only** | Manual v1 | 40.6% | 62.5% | 68.8% | 0.5104 | 1.5 ms |
| **BM25 Only** | LangChain | 40.6% | 62.5% | 68.8% | 0.5052 | **1.2 ms** |
| **Hybrid (RRF)** | Manual v1 | 43.8% | 87.5% | 96.9% | 0.6557 | 39.1 ms |
| **Hybrid (RRF)** | LangChain | **46.9%** | 78.1% | 93.8% | 0.6505 | **25.5 ms** |
| **Hybrid + Rerank** | Manual v1 | 71.9% | 90.6% | 96.9% | 0.8177 | 2587.5 ms |
| **Hybrid + Rerank** | LangChain | **78.1%** | **90.6%** | **96.9%** | **0.8510** | **2135.8 ms** |

### Category Breakdown (Current LangChain)

| Category | Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|---|:---:|:---:|:---:|:---:|
| **Conceptual / Natural Language** *(22 queries)* | **Hit@5** | 86.4% | 54.5% | 90.9% | **95.5%** |
| | **MRR** | 0.7462 | 0.3333 | 0.5371 | **0.8364** |
| **Exact Symbol / Identifier** *(10 queries)* | **Hit@5** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| | **MRR** | 0.7667 | 0.8833 | **0.9000** | 0.8833 |

---

## Key Architecture & Design Decisions

| Decision | Rationale |
|----------|-----------|
| **LangChain components for standard RAG** | Replaced custom Chroma client, BM25 persistence, RRF math, cross-encoder loop, and REST/Ollama wrappers with `HuggingFaceEmbeddings`, `Chroma`, `BM25Retriever`, `EnsembleRetriever`, `CrossEncoderReranker`, `ChatPromptTemplate`, `ChatOllama`/`ChatGoogleGenerativeAI`, and LCEL (`-825` source lines). |
| **Kept custom AST chunker & file loader** | Python `ast` preserves function/method/class boundaries, decorators, parent class, and exact line numbers while outputting `Chunk(Document)`. Non-Python files fall back to 50-line overlapping windows. |
| **Metadata header in `Document.page_content`** | Prepending `# <type> <parent>.<symbol> \| File: <path>` during indexing restored structural context for embeddings, BM25, and the cross-encoder (lifting Vector Hit@5 from 84.4% to 90.6% and Rerank MRR from 0.7589 to 0.8510). Stripped cleanly before UI display and LLM prompt formatting so line metadata stays exact. |
| **Identifier-aware tokenization + zero-score filter** | `tokenize_code` splits `snake_case`/`camelCase` identifiers as `BM25Retriever`'s `preprocess_func`; `_PositiveBM25Retriever` drops `score <= 0` documents so out-of-vocabulary queries return `[]`. |
| **Repo fingerprint caching** | Git SHA or content MD5 in `data/<repo>_meta.json` skips re-indexing unchanged repos. |
| **Citation post-verification & refusal** | Strict `ChatPromptTemplate` with positive/negative few-shot examples + regex verification of every `file.py:start-end` citation against retrieved `Document.metadata`. |
| **8 GB RAM Windows budget** | Batch embedding size 32, lazy-loaded singletons, max 4 chunks (2,500 chars each) to the LLM, `num_ctx=4096`. |
