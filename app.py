"""
CodeCompass — Web Application (Streamlit UI) powered by LangChain.

Features:
  - Index any GitHub URL or local repository (AST chunking + Chroma + BM25).
  - Switch between Hybrid (RRF), Hybrid + Cross-Encoder Rerank, Vector, and BM25.
  - Switch between local Ollama (qwen2.5-coder:3b) and Gemini Cloud API (.env).
  - Stream answers via the LangChain LCEL chain with post-stream citation verification
    and expandable syntax-highlighted source code snippets.
"""

from pathlib import Path

import streamlit as st
from langchain_core.documents import Document

from src.config import GEMINI_MODEL, LLM_MODEL, TOP_K_FINAL
from src.generation.chain import (
    check_llm_health,
    get_chat_model,
    stream_answer,
    verify_all_citations,
)
from src.ingestion.indexer import get_docs_path, index_repo
from src.ingestion.loader import repo_name_from_url
from src.retrieval.retrieval import clear_retriever_cache, get_retriever


# ── Constants & Helpers ──────────────────────────────────────────────────────

MODE_MAP: dict[str, str] = {
    "Hybrid (Vector + BM25 via RRF)": "hybrid",
    "Hybrid + Rerank": "hybrid_rerank",
    "Vector Only": "vector",
    "BM25 Only": "bm25",
}


def resolve_repo_name(source: str) -> str:
    """Extract a clean repo name from a GitHub URL or local directory path."""
    source_clean = source.strip()
    if source_clean.startswith(("http://", "https://")):
        return repo_name_from_url(source_clean)
    return Path(source_clean).name


def check_repo_indexed(repo_name: str) -> bool:
    """Fast check for whether serialized Documents exist for this repo."""
    return get_docs_path(repo_name).exists()


def retrieve_by_mode(
    query: str,
    repo_name: str,
    mode: str,
    top_k: int = TOP_K_FINAL,
) -> list[Document]:
    """Dispatch a query to the selected LangChain retriever mode."""
    mode_key = MODE_MAP.get(mode, "hybrid")
    retriever = get_retriever(repo_name, mode=mode_key, top_k=top_k)
    return retriever.invoke(query)


def render_citations_and_snippets(
    docs: list[Document],
    valid: list[str],
    hallucinated: list[str],
) -> None:
    """Display verified/unverified citation badges and expandable code snippets."""
    if valid:
        st.markdown("**Verified Citations:** " + ", ".join(f"`{c}`" for c in valid))
    if hallucinated:
        st.warning("**Unverified Citations:** " + ", ".join(f"`{c}`" for c in hallucinated))

    if not docs:
        return

    with st.expander(f"📄 View Retrieved Code Snippets ({len(docs)})", expanded=False):
        for idx, doc in enumerate(docs, 1):
            m = doc.metadata
            sym = m.get("symbol_name") or "(block)"
            parent = f" in {m['parent_class']}" if m.get("parent_class") else ""
            file_line = f"{m.get('file_path', 'unknown')}:{m.get('start_line', 0)}-{m.get('end_line', 0)}"
            chunk_type = m.get("chunk_type", "code")
            lang = m.get("language") or "python"

            st.markdown(f"**Snippet [{idx}]** — `{file_line}` | `{chunk_type}` **{sym}**{parent}")
            st.code(doc.page_content, language=lang, line_numbers=True)


# ── Page Configuration & Session State ───────────────────────────────────────

st.set_page_config(
    page_title="CodeCompass — Codebase Assistant",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_indexing" not in st.session_state:
    st.session_state.is_indexing = False


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧭 CodeCompass")
    st.caption("Codebase Onboarding Assistant (LangChain RAG)")
    st.divider()

    # 1. LLM Provider Switch & Health Check
    st.subheader("🤖 LLM Backend")
    provider_option = st.selectbox(
        "Provider:",
        options=["Ollama (Local)", "Gemini (Cloud API)"],
        index=0,
        help="Switch between local Ollama and Google Gemini API (configured in .env).",
    )
    chosen_provider = "gemini" if "Gemini" in provider_option else "ollama"
    active_model = GEMINI_MODEL if chosen_provider == "gemini" else LLM_MODEL

    is_healthy, health_msg = check_llm_health(provider=chosen_provider)
    if is_healthy:
        st.success(f"**Connected**: `{active_model}`")
    else:
        st.error("**Provider Offline or Unconfigured**")
        st.info(health_msg)

    st.divider()

    # 2. Repository Indexing
    st.subheader("📦 Repository")
    repo_input = st.text_input(
        "GitHub URL or Local Path:",
        value="data/flask-realworld-example-app",
        help="Paste a GitHub URL or local folder path.",
    )
    current_repo_name = resolve_repo_name(repo_input)
    is_indexed = check_repo_indexed(current_repo_name)

    if is_indexed:
        st.caption(f"Status: ✅ **Indexed** (`{current_repo_name}`)")
    else:
        st.caption(f"Status: ⚠️ **Not indexed yet** (`{current_repo_name}`)")

    col_idx1, col_idx2 = st.columns([2, 1])
    force_reindex = col_idx2.checkbox("Force", value=False, help="Re-index from scratch")
    index_btn = col_idx1.button(
        "⚡ Index Repo",
        use_container_width=True,
        disabled=st.session_state.is_indexing,
    )

    if index_btn:
        st.session_state.is_indexing = True
        with st.spinner(f"Indexing '{current_repo_name}'..."):
            try:
                result = index_repo(repo_input, force_reindex=force_reindex)
                clear_retriever_cache()
                st.session_state.is_indexing = False
                if result.get("skipped"):
                    st.info(f"Using cached index: **{result['total_chunks']} chunks**.")
                else:
                    st.success(
                        f"Indexed **{result['total_files']} files** into "
                        f"**{result['total_chunks']} chunks** in {result['duration_seconds']}s!"
                    )
                st.rerun()
            except Exception as exc:
                st.session_state.is_indexing = False
                st.error(f"Indexing error: {exc}")

    st.divider()

    # 3. Retrieval Settings
    st.subheader("🔍 Retrieval Settings")
    search_mode = st.radio(
        "Search Engine Mode:",
        options=list(MODE_MAP.keys()),
        index=0,
        help="Compare Hybrid RRF, Cross-Encoder Reranking, Vector Only, and BM25 Only.",
    )
    top_k_chunks = st.slider(
        "Chunks fed to LLM:",
        min_value=1,
        max_value=TOP_K_FINAL,
        value=TOP_K_FINAL,
        help="Capped at 4 chunks to fit comfortably within 8 GB RAM.",
    )

    st.divider()

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Area ───────────────────────────────────────────────────────────

st.title("Ask Your Codebase Anything")
st.markdown(
    "Explore architecture, locate functions, and trace request flows. "
    "Every answer cites exact file paths and line numbers."
)

if not st.session_state.messages:
    st.info(
        "💡 **Try asking:**\n"
        "- *'Where is user authentication handled?'*\n"
        "- *'How does an article get created and saved to the database?'*\n"
        "- *'Where is jwt_optional used and what does it do?'*\n"
        "- *'How does the application process payments?'* (Tests grounded refusal)"
    )

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_citations_and_snippets(
                msg.get("docs", []),
                msg.get("valid_citations", []),
                msg.get("hallucinated_citations", []),
            )


# ── Chat Input & LCEL Streaming ──────────────────────────────────────────────

user_query = st.chat_input(
    "Ask a question about this repository...",
    disabled=st.session_state.is_indexing,
)

if user_query:
    if not is_indexed:
        st.warning(f"Please click **⚡ Index Repo** to index `{current_repo_name}` first.")
    elif not is_healthy:
        st.error(f"Cannot generate answer: {health_msg}")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            mode_key = MODE_MAP.get(search_mode, "hybrid")
            retriever = get_retriever(current_repo_name, mode=mode_key, top_k=top_k_chunks)
            llm = get_chat_model(provider=chosen_provider)

            try:
                with st.spinner(f"Searching `{current_repo_name}` via {search_mode}..."):
                    token_stream, retrieved_docs = stream_answer(user_query, retriever, llm=llm)
                full_answer = st.write_stream(token_stream)
            except Exception as exc:
                st.error(f"Generation error: {exc}")
                full_answer = f"Error during generation: {exc}"
                retrieved_docs = []

            valid, hallucinated = verify_all_citations(full_answer, retrieved_docs)
            render_citations_and_snippets(retrieved_docs, valid, hallucinated)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "docs": retrieved_docs,
                "valid_citations": valid,
                "hallucinated_citations": hallucinated,
            })
