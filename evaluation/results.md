# CodeCompass — Retrieval Evaluation Report

> **Target Repository:** `flask-realworld-example-app`  
> **Benchmark Dataset:** 32 ground-truth questions  
> **Retrieval Configurations:** Vector-Only, BM25-Only, Hybrid (RRF), Hybrid + Cross-Encoder Rerank  

---

## 1. Overall Performance Comparison

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|:---:|:---:|:---:|:---:|
| **Hit@1** | 62.5% | 40.6% | **43.8%** | **71.9%** |
| **Hit@3** | 87.5% | 62.5% | **87.5%** | **90.6%** |
| **Hit@5** | 90.6% | 68.8% | **96.9%** | **96.9%** |
| **MRR (Mean Reciprocal Rank)** | 0.7474 | 0.5104 | **0.6557** | **0.8177** |
| **Average Latency (ms)** | 37.3 ms | 1.5 ms | 39.1 ms | 2587.5 ms |

---

## 2. Category Breakdown Analysis

### A. Conceptual & Architectural Queries (Natural Language)
*Queries asking where components live or how flows work (e.g. "Where is the Flask application factory defined?", "How does password hashing work?").*

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|:---:|:---:|:---:|:---:|
| **Hit@5** | 86.4% | 54.5% | **95.5%** | 95.5% |
| **MRR** | 0.7008 | 0.3333 | **0.5674** | 0.8333 |

### B. Exact Identifier & Symbol Queries
*Queries naming exact programming symbols (e.g. "`jwt_identity`", "`SurrogatePK`", "`is_following`").*

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|:---:|:---:|:---:|:---:|
| **Hit@5** | 100.0% | 100.0% | **100.0%** | 100.0% |
| **MRR** | 0.8500 | 0.9000 | **0.8500** | 0.7833 |

---

## 3. Analysis & Observations

1. **Hybrid vs. Vector Only:**
   - **Hit@5:** Hybrid retrieval achieved a higher Hit@5 (**96.9%**) than Vector Only (90.6%) and BM25 Only (68.8%). Combining dense and sparse retrieval helped surface target files that vector search missed (such as configuration classes).
   - **Hit@1 and MRR:** However, Vector Only had a higher Hit@1 (62.5% vs. 43.8%) and higher MRR (0.7474 vs. 0.6557) than Hybrid. Reciprocal Rank Fusion frequently placed the correct chunk within the top 3 to 5 results, but vector search placed it directly at Rank 1 more often when it matched.

2. **Reranker Trade-offs:**
   - **Ranking Precision:** Hybrid + Rerank achieved the highest MRR (**0.8177**) and the highest Hit@1 (**71.9%**), successfully placing target snippets at Rank 1 in most queries.
   - **Latency:** It also had the highest latency (**2587.5 ms** vs. 39.1 ms for Hybrid and 37.3 ms for Vector).

3. **Scope and Dataset Limits:**
   - These findings are based on a small evaluation set of 32 questions on a single repository (`flask-realworld-example-app`). Results on larger repositories or other programming languages may differ.
