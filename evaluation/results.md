# CodeCompass — Retrieval Evaluation Report

> **Target Repository:** `flask-realworld-example-app`  
> **Benchmark Dataset:** 32 ground-truth questions  
> **Retrieval Configurations:** Vector-Only, BM25-Only, Hybrid (RRF), Hybrid + Cross-Encoder Rerank  

---

## 1. Overall Performance Comparison

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|:---:|:---:|:---:|:---:|
| **Hit@1** | 50.0% | 28.1% | 34.4% | **65.6%** |
| **Hit@3** | 65.6% | 53.1% | 65.6% | **87.5%** |
| **Hit@5** | 84.4% | 65.6% | 81.2% | **93.8%** |
| **MRR (Mean Reciprocal Rank)** | 0.6099 | 0.4083 | 0.5255 | **0.7589** |
| **Average Latency (ms)** | 22.1 ms | 1.0 ms | 22.9 ms | 1960.4 ms |

---

## 2. Category Breakdown Analysis

### A. Conceptual & Architectural Queries (Natural Language)
*Queries asking where components live or how flows work (e.g. "Where is the Flask application factory defined?", "How does password hashing work?").*

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|:---:|:---:|:---:|:---:|
| **Hit@5** | 77.3% | 50.0% | 72.7% | **90.9%** |
| **MRR** | 0.5583 | 0.2417 | 0.4008 | **0.7629** |

### B. Exact Identifier & Symbol Queries
*Queries naming exact programming symbols (e.g. "`jwt_identity`", "`SurrogatePK`", "`is_following`").*

| Metric | Vector Only | BM25 Only | Hybrid (RRF) | Hybrid + Rerank |
|---|:---:|:---:|:---:|:---:|
| **Hit@5** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |
| **MRR** | 0.7233 | 0.7750 | **0.8000** | 0.7500 |

---

## 3. Analysis & Observations

1. **Hybrid (RRF) vs. Vector Only and BM25 Only:**
   - **Hit@5:** Hybrid (RRF) reached **81.2%**, which is lower than Vector Only (84.4%) and higher than BM25 Only (65.6%).
   - **Hit@1 and MRR:** Hybrid (RRF) reached 34.4% Hit@1 and 0.5255 MRR, which is lower than Vector Only (50.0% Hit@1, 0.6099 MRR).

2. **Reranker Trade-offs:**
   - **Ranking Precision & Recall:** Hybrid + Rerank achieved Hit@1 of **65.6%**, Hit@5 of **93.8%**, and MRR of **0.7589** (compared to Hybrid RRF at 34.4% Hit@1, 81.2% Hit@5, and 0.5255 MRR).
   - **Latency:** Hybrid + Rerank had the highest average latency (**1960.4 ms** vs. 22.9 ms for Hybrid, 22.1 ms for Vector, and 1.0 ms for BM25).

3. **Scope and Dataset Limits:**
   - These findings are based on a small evaluation set of 32 questions on a single repository (`flask-realworld-example-app`). Results on larger repositories or other programming languages may differ.
