# CodeCompass — Interview Demo Plan & Presentation Script

This document provides a structured, 3-minute walkthrough script for showcasing **CodeCompass** during AI engineering interviews or portfolio recordings.

---

## 🎯 30-Second Elevator Pitch

> *"CodeCompass is an onboarding assistant that lets engineers ask questions about any codebase and receive grounded answers with verified file and line citations. Rather than using naive character-chunking and standard vector search, it combines Python AST-aware chunking, dense embeddings, BM25 keyword matching with code identifier splitting, Reciprocal Rank Fusion, and an optional Cross-Encoder reranker. On an 8 GB RAM laptop running a local 3B model, it delivers a 96.9% Hit@5 retrieval rate on real-world queries."*

---

## ⏱️ 3-Minute Live Demo Script

### Step 1: Launch the Application (0:00 – 0:30)

1. Start Ollama and verify the model:
   ```powershell
   ollama list
   # Model qwen2.5-coder:3b should be present
   ```
2. Start the Streamlit application:
   ```powershell
   cd codecompass
   .\venv\Scripts\streamlit.exe run app.py
   ```
3. Open `http://localhost:8501` in your browser.
4. **Talking Point:**
   > *"Notice the sidebar status check. The system connects to our local Ollama backend running `qwen2.5-coder:3b` with a 4096 context window. It also features a zero-dependency fallback to Google's Gemini Cloud API for hosted environments."*

---

### Step 2: Indexing & Intelligent Caching (0:30 – 1:00)

1. In the sidebar, point out the repository input: `data/flask-realworld-example-app`.
2. Click **⚡ Index Repo**.
3. Point out that the index loads in less than 3 seconds.
4. **Talking Point:**
   > *"CodeCompass calculates a repository content fingerprint based on Git commits or file modification hashes. When the codebase is unchanged, it skips redundant AST parsing and vector calculations, loading the index instantly."*

---

### Step 3: Natural Language / Architectural Query (1:00 – 1:45)

1. In the chat input, ask:
   ```text
   Where is user registration handled, and what fields are required?
   ```
2. Watch the tokens stream in real time.
3. Observe the response:
   - Identifies `conduit/user/views.py`.
   - Explains that `create_user` endpoint validates username, email, and password.
4. Point out the **Verified Citations** badge:
   `conduit/user/views.py:12-25`
5. Expand the **View Retrieved Code Snippets** accordion to show the syntax-highlighted AST chunk.
6. **Talking Point:**
   > *"The LLM streams tokens immediately, lowering perceived latency. Notice the green citation badge: CodeCompass programmatically parses citations in the generated response and validates that they genuinely overlap with the retrieved AST chunks, catching hallucinations before they reach the user."*

---

### Step 4: Exact Identifier Query & Search Mode Comparison (1:45 – 2:30)

1. Switch search mode in the sidebar from **Vector Only** to **BM25 Only**, then to **Hybrid (Vector + BM25 via RRF)**.
2. Ask:
   ```text
   Where is SurrogatePK defined and what is its purpose?
   ```
3. **Talking Point:**
   > *"This highlights why Hybrid Search with RRF is critical for code. In semantic vector search alone, exact identifier names like `SurrogatePK` can get lost among unrelated database snippets. BM25 with identifier-aware tokenization matches the exact class symbol, while vector search captures the architectural intent. Reciprocal Rank Fusion combines their rank positions ($k=60$) without needing fragile score calibration."*

---

### Step 5: Grounded Refusal / Hallucination Resistance (2:30 – 3:00)

1. Ask a question outside the codebase's scope:
   ```text
   How does this application handle Stripe credit card payments?
   ```
2. Observe the grounded refusal response:
   > *"I couldn't find this in the codebase. The retrieved code context does not contain any Stripe payment processing logic."*
3. **Talking Point:**
   > *"In codebase assistants, false answers are worse than no answers. We enforce strict prompt grounding with few-shot negative examples so the model politely refuses when context is insufficient."*

---

## 📊 Key Numbers to Recite from Memory

- **96.9% Hit@5:** The ground-truth code chunk is retrieved in the top 5 candidates for 31 out of 32 benchmark questions.
- **MRR 0.8177:** With the cross-encoder reranker enabled, the exact target code lands at Rank #1 in **71.9%** of queries.
- **Hybrid vs Vector Trade-off:** Hybrid achieves higher Hit@5 (96.9% vs 90.6%) by finding items vector search missed, but has lower Hit@1 (43.8% vs 62.5%) and lower MRR (0.6557 vs 0.7474) because fusion often places results across ranks 2 to 5.
- **39 ms vs 2.6 s:** Steady-state Hybrid RRF operates in ~39 ms (Vector 37 ms + BM25 1.5 ms); adding the cross-encoder reranker elevates Rank #1 placement at the cost of ~2.6 s of CPU inference.
- **8 GB RAM:** Engineered to stay well within memory limits by batching embeddings in groups of 32 and running lazy-loaded singleton models.

