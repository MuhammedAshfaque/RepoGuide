# CodeCompass — Retrieval Evaluation Report

> **Target Repository:** `flask-realworld-example-app`  
> **Benchmark Dataset:** 32 ground-truth questions  
> **Retrieval Configurations:** Vector-Only, BM25-Only, Hybrid (RRF), Hybrid + Cross-Encoder Rerank  

---

## 1. Overall Performance Comparison

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|:---:|:---:|:---:|:---:|
| **Hit@1** | 65.6% | 40.6% | 46.9% | **78.1%** |
| **Hit@3** | 87.5% | 62.5% | 78.1% | **90.6%** |
| **Hit@5** | 90.6% | 68.8% | 93.8% | **96.9%** |
| **MRR (Mean Reciprocal Rank)** | 0.7526 | 0.5052 | 0.6505 | **0.8510** |
| **Average Latency (ms)** | 26.2 ms | 1.2 ms | 26.1 ms | 2187.5 ms |

---

## 2. Category Breakdown Analysis

### A. Conceptual & Architectural Queries (Natural Language)
*Queries asking where components live or how flows work (e.g. "Where is the Flask application factory defined?", "How does password hashing work?").*

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|:---:|:---:|:---:|:---:|
| **Hit@5** | 86.4% | 54.5% | 90.9% | **95.5%** |
| **MRR** | 0.7462 | 0.3333 | 0.5371 | **0.8364** |

### B. Exact Identifier & Symbol Queries
*Queries naming exact programming symbols (e.g. "`jwt_identity`", "`SurrogatePK`", "`is_following`").*

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|:---:|:---:|:---:|:---:|
| **Hit@5** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **MRR** | 0.7667 | 0.8833 | **0.9000** | 0.8833 |

---

## 3. Analysis & Observations

1. **Hybrid (RRF) vs. Vector Only and BM25 Only:**
   - **Hit@5:** Hybrid (RRF) reached **93.8%**, which is higher than Vector Only (90.6%) and higher than BM25 Only (68.8%).
   - **Hit@1 and MRR:** Hybrid (RRF) reached 46.9% Hit@1 and 0.6505 MRR, which is lower than Vector Only (65.6% Hit@1, 0.7526 MRR).

2. **Reranker Trade-offs:**
   - **Ranking Precision & Recall:** Hybrid + Rerank achieved Hit@1 of **78.1%**, Hit@5 of **96.9%**, and MRR of **0.8510** (compared to Hybrid RRF at 46.9% Hit@1, 93.8% Hit@5, and 0.6505 MRR).
   - **Latency:** Hybrid + Rerank had the highest average latency (**2187.5 ms** vs. 26.1 ms for Hybrid, 26.2 ms for Vector, and 1.2 ms for BM25).

3. **Scope and Dataset Limits:**
   - These findings are based on a small evaluation set of 32 questions on a single repository (`flask-realworld-example-app`). Results on larger repositories or other programming languages may differ.
