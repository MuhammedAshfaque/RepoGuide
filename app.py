"""
CodeCompass — Web Application (Streamlit UI).

A code onboarding assistant using RAG:
  - Ingests repositories via GitHub URL or local path.
  - Hybrid retrieval: Vector embeddings (ChromaDB) + BM25 keyword matching via RRF.
  - Local LLM generation using Ollama (qwen2.5-coder:3b) with token streaming.
  - Post-generation citation audit & expandable syntax-highlighted code viewers.
"""

from pathlib import Path
from typing import Any

import streamlit as st

from src.config import DATA_DIR, TOP_K_FINAL, LLM_MODEL, get_collection_name
from src.ingestion.loader import repo_name_from_url
from src.ingestion.indexer import index_repo
from src.retrieval.hybrid import hybrid_search
from src.retrieval.vector_search import vector_search
from src.retrieval.keyword_search import bm25_search, get_bm25_index_path
from src.generation.llm import get_llm_provider
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from src.generation.answerer import verify_all_citations


# ── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="CodeCompass — Codebase Assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── State Initialization ─────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_indexing" not in st.session_state:
    st.session_state.is_indexing = False


# ── Helper Functions ─────────────────────────────────────────────────────────

def resolve_repo_name(source: str) -> str:
    """Extract clean repo name from URL or local directory path."""
    source_clean = source.strip()
    if source_clean.startswith("http://") or source_clean.startswith("https://"):
        return repo_name_from_url(source_clean)
    return Path(source_clean).name


def check_repo_indexed(repo_name: str) -> bool:
    """Check if both BM25 index and ChromaDB collection exist for repo."""
    bm25_path = get_bm25_index_path(repo_name)
    return bm25_path.exists()


def retrieve_by_mode(
    query: str,
    repo_name: str,
    mode: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """Dispatch query to selected retrieval engine."""
    if mode == "Hybrid (Vector + BM25 via RRF)":
        return hybrid_search(query, repo_name, top_k_retrieval=20, top_k_final=top_k)
    elif mode == "Hybrid + Rerank":
        from src.retrieval.reranker import hybrid_search_with_rerank
        return hybrid_search_with_rerank(query, repo_name, top_k_retrieval=20, top_k_final=top_k)
    elif mode == "Vector Only":
        return vector_search(query, repo_name, top_k=top_k)
    elif mode == "BM25 Only":
        return bm25_search(query, repo_name, top_k=top_k)
    else:
        return hybrid_search(query, repo_name, top_k_final=top_k)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧭 CodeCompass")
    st.caption("A Codebase Onboarding Assistant using RAG")
    st.divider()

    # 1. LLM Backend Status Check & Provider Selection
    st.subheader("🤖 LLM Backend")
    provider_option = st.selectbox(
        "Provider:",
        options=["Ollama (Local)", "Gemini (Cloud API)"],
        index=0,
        help="Select local Ollama or Google Gemini Cloud API (configured via .env)",
    )
    chosen_provider = "gemini" if "Gemini" in provider_option else "ollama"

    llm = get_llm_provider(provider_name=chosen_provider)
    is_healthy, health_msg = llm.check_health()
    if is_healthy:
        st.success(f"**Connected**: `{getattr(llm, 'model', chosen_provider)}`")
    else:
        st.error("**Provider Offline or Unconfigured**")
        st.info(health_msg)

    st.divider()

    # 2. Repository Configuration & Indexing
    st.subheader("📦 Repository")
    default_repo_path = "data/flask-realworld-example-app"
    repo_input = st.text_input(
        "GitHub URL or Local Path:",
        value=default_repo_path,
        help="Paste a GitHub URL (e.g. https://github.com/user/repo) or local folder path.",
    )

    current_repo_name = resolve_repo_name(repo_input)
    is_indexed = check_repo_indexed(current_repo_name)

    if is_indexed:
        st.caption(f"Status: ✅ **Indexed** (`{current_repo_name}`)")
    else:
        st.caption(f"Status: ⚠️ **Not indexed yet** (`{current_repo_name}`)")

    col_idx1, col_idx2 = st.columns([2, 1])
    force_reindex = col_idx2.checkbox("Force", value=False, help="Re-index from scratch")
    index_btn = col_idx1.button("⚡ Index Repo", use_container_width=True, disabled=st.session_state.is_indexing)

    if index_btn:
        st.session_state.is_indexing = True
        with st.spinner(f"Indexing '{current_repo_name}' (parsing AST & computing embeddings)..."):
            try:
                result = index_repo(repo_input, force_reindex=force_reindex)
                st.session_state.is_indexing = False
                if result.get("skipped"):
                    st.info(f"Using existing index: **{result['total_chunks']} chunks** loaded.")
                else:
                    st.success(
                        f"Indexed **{result['total_files']} files** into **{result['total_chunks']} chunks** in {result['duration_seconds']}s!"
                    )
                st.rerun()
            except Exception as e:
                st.session_state.is_indexing = False
                st.error(f"Indexing error: {e}")

    st.divider()

    # 3. Retrieval Settings
    st.subheader("🔍 Retrieval Settings")
    search_mode = st.radio(
        "Search Engine Mode:",
        options=[
            "Hybrid (Vector + BM25 via RRF)",
            "Hybrid + Rerank",
            "Vector Only",
            "BM25 Only",
        ],
        index=0,
        help="Compare semantic search vs exact identifier keyword matching vs cross-encoder reranking.",
    )

    top_k_chunks = st.slider(
        "Chunks fed to LLM:",
        min_value=1,
        max_value=8,
        value=TOP_K_FINAL,
        help="Number of retrieved code snippets provided in prompt context.",
    )

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Content Area ────────────────────────────────────────────────────────

st.title("Ask Your Codebase Anything")
st.markdown(
    "Explore architecture, locate functions, and trace request flows. "
    "Every answer cites exact file paths and line numbers."
)

# Display empty state welcome banner if no messages
if not st.session_state.messages:
    with st.container():
        st.info(
            "💡 **Try asking:**\n"
            "- *'Where is user authentication handled?'*\n"
            "- *'How does an article get created and saved to the database?'*\n"
            "- *'Where is jwt_optional used and what does it do?'*\n"
            "- *'How does the application process payments?'* (Tests grounded refusal)"
        )

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # If assistant response has citation / snippet metadata
        if msg.get("role") == "assistant" and msg.get("chunks"):
            valid = msg.get("valid_citations", [])
            hallucinated = msg.get("hallucinated_citations", [])

            if valid:
                st.markdown(f"**Verified Citations:** " + ", ".join(f"`{c}`" for c in valid))
            if hallucinated:
                st.warning(f"**Unverified Citations:** " + ", ".join(f"`{c}`" for c in hallucinated))

            with st.expander(f"📄 View Retrieved Code Snippets ({len(msg['chunks'])})", expanded=False):
                for idx, chunk in enumerate(msg["chunks"], 1):
                    sym = chunk.get("symbol_name") or "(block)"
                    parent = f" in {chunk['parent_class']}" if chunk.get("parent_class") else ""
                    file_line = f"{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}"
                    score_info = f"score: {chunk.get('score', 0.0):.4f}" if "score" in chunk else ""

                    st.markdown(f"**Snippet [{idx}]** — `{file_line}` | `{chunk.get('chunk_type', 'code')}` **{sym}**{parent} ({score_info})")
                    lang = chunk.get("language") or "python"
                    st.code(chunk.get("content", ""), language=lang, line_numbers=True)


# ── Chat Input & Streaming Execution ─────────────────────────────────────────

user_query = st.chat_input("Ask a question about this repository...", disabled=st.session_state.is_indexing)

if user_query:
    if not is_indexed:
        st.warning(f"Please click **⚡ Index Repo** in the sidebar to index `{current_repo_name}` before querying.")
    elif not is_healthy:
        st.error(f"Cannot generate answer: {health_msg}")
    else:
        # 1. Display User Message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 2. Assistant Turn
        with st.chat_message("assistant"):
            # Step A: Retrieve Chunks
            with st.spinner(f"Searching `{current_repo_name}` via {search_mode}..."):
                retrieved_chunks = retrieve_by_mode(
                    query=user_query,
                    repo_name=current_repo_name,
                    mode=search_mode,
                    top_k=top_k_chunks,
                )

            if not retrieved_chunks:
                fallback_text = "I couldn't find this in the codebase. No relevant code chunks were retrieved."
                st.markdown(fallback_text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": fallback_text,
                    "chunks": [],
                    "valid_citations": [],
                    "hallucinated_citations": [],
                })
            else:
                # Step B: Assemble Prompt & Stream Response
                user_prompt = build_user_prompt(user_query, retrieved_chunks)

                try:
                    stream_gen = llm.stream(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
                    full_answer = st.write_stream(stream_gen)
                except Exception as exc:
                    st.error(f"Generation error: {exc}")
                    full_answer = f"Error during generation: {exc}"

                # Step C: Citation Verification Audit
                valid, hallucinated = verify_all_citations(full_answer, retrieved_chunks)

                if valid:
                    st.markdown(f"**Verified Citations:** " + ", ".join(f"`{c}`" for c in valid))
                if hallucinated:
                    st.warning(f"**Unverified Citations:** " + ", ".join(f"`{c}`" for c in hallucinated))

                # Step D: Expandable Source Code Viewer
                with st.expander(f"📄 View Retrieved Code Snippets ({len(retrieved_chunks)})", expanded=False):
                    for idx, chunk in enumerate(retrieved_chunks, 1):
                        sym = chunk.get("symbol_name") or "(block)"
                        parent = f" in {chunk['parent_class']}" if chunk.get("parent_class") else ""
                        file_line = f"{chunk['file_path']}:{chunk['start_line']}-{chunk['end_line']}"
                        if "rerank_score" in chunk:
                            score_info = f"rerank score: {chunk['rerank_score']:.4f}"
                        elif "score" in chunk:
                            score_info = f"score: {chunk['score']:.4f}"
                        else:
                            score_info = ""

                        st.markdown(f"**Snippet [{idx}]** — `{file_line}` | `{chunk.get('chunk_type', 'code')}` **{sym}**{parent} ({score_info})")
                        lang = chunk.get("language") or "python"
                        st.code(chunk.get("content", ""), language=lang, line_numbers=True)

                # Save assistant turn to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "chunks": retrieved_chunks,
                    "valid_citations": valid,
                    "hallucinated_citations": hallucinated,
                })

