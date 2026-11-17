"""CodeCompass Streamlit UI: index a repo, pick a search mode & LLM, and stream cited answers."""

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
from src.ingestion.indexer import get_meta_path, index_repo, strip_metadata_header
from src.ingestion.loader import repo_name_from_url
from src.retrieval.retrieval import clear_retriever_cache, get_retriever

MODE_MAP: dict[str, str] = {
    "Hybrid (Vector + BM25 via RRF)": "hybrid",
    "Hybrid + Rerank": "hybrid_rerank",
    "Vector Only": "vector",
    "BM25 Only": "bm25",
}


def resolve_repo_name(source: str) -> str:
    """Extract a clean repository name from a GitHub URL or local folder path."""
    s = source.strip()
    return repo_name_from_url(s) if s.startswith(("http://", "https://")) else Path(s).name


def check_repo_indexed(repo_name: str) -> bool:
    """Fast check for whether index metadata exists for this repository."""
    return get_meta_path(repo_name).exists()


def render_citations_and_snippets(docs: list[Document], valid: list[str], hallucinated: list[str]) -> None:
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
            loc = f"{m.get('file_path', 'unknown')}:{m.get('start_line', 0)}-{m.get('end_line', 0)}"
            st.markdown(f"**Snippet [{idx}]** — `{loc}` | `{m.get('chunk_type', 'code')}` **{sym}**{parent}")
            st.code(strip_metadata_header(doc.page_content), language=m.get("language") or "python", line_numbers=True)


# ── Page Setup & State ───────────────────────────────────────────────────────

st.set_page_config(page_title="CodeCompass — Codebase Assistant", page_icon="🧭", layout="wide")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("is_indexing", False)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧭 CodeCompass")
    st.caption("Codebase Onboarding Assistant (LangChain RAG)")
    st.divider()

    st.subheader("🤖 LLM Backend")
    provider_label = st.selectbox("Provider:", ["Ollama (Local)", "Gemini (Cloud API)"])
    provider = "gemini" if "Gemini" in provider_label else "ollama"
    active_model = GEMINI_MODEL if provider == "gemini" else LLM_MODEL

    is_healthy, health_msg = check_llm_health(provider=provider)
    if is_healthy:
        st.success(f"**Connected**: `{active_model}`")
    else:
        st.error("**Provider Offline or Unconfigured**")
        st.info(health_msg)

    st.divider()

    st.subheader("📦 Repository")
    repo_input = st.text_input("GitHub URL or Local Path:", value="data/flask-realworld-example-app")
    repo_name = resolve_repo_name(repo_input)
    is_indexed = check_repo_indexed(repo_name)
    st.caption(f"Status: {'✅ **Indexed**' if is_indexed else '⚠️ **Not indexed yet**'} (`{repo_name}`)")

    col1, col2 = st.columns([2, 1])
    force_reindex = col2.checkbox("Force", value=False)
    if col1.button("⚡ Index Repo", use_container_width=True, disabled=st.session_state.is_indexing):
        st.session_state.is_indexing = True
        with st.spinner(f"Indexing '{repo_name}'..."):
            try:
                res = index_repo(repo_input, force_reindex=force_reindex)
                clear_retriever_cache()
                st.session_state.is_indexing = False
                if res.get("skipped"):
                    st.info(f"Using cached index: **{res['total_chunks']} chunks**.")
                else:
                    st.success(f"Indexed **{res['total_files']} files** into **{res['total_chunks']} chunks** in {res['duration_seconds']}s!")
                st.rerun()
            except Exception as exc:
                st.session_state.is_indexing = False
                st.error(f"Indexing error: {exc}")

    st.divider()

    st.subheader("🔍 Retrieval Settings")
    search_mode = st.radio("Search Engine Mode:", options=list(MODE_MAP.keys()), index=0)
    top_k = st.slider("Chunks fed to LLM:", min_value=1, max_value=TOP_K_FINAL, value=TOP_K_FINAL)

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Area ───────────────────────────────────────────────────────────

st.title("Ask Your Codebase Anything")
st.markdown("Explore architecture, locate functions, and trace request flows with verified file and line citations.")

if not st.session_state.messages:
    st.info(
        "💡 **Try asking:**\n"
        "- *'Where is user authentication handled?'*\n"
        "- *'How does an article get created and saved to the database?'*\n"
        "- *'Where is jwt_optional used and what does it do?'*\n"
        "- *'How does the application process payments?'* (Tests grounded refusal)"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_citations_and_snippets(msg.get("docs", []), msg.get("valid_citations", []), msg.get("hallucinated_citations", []))

user_query = st.chat_input("Ask a question about this repository...", disabled=st.session_state.is_indexing)
if user_query:
    if not is_indexed:
        st.warning(f"Please click **⚡ Index Repo** to index `{repo_name}` first.")
    elif not is_healthy:
        st.error(f"Cannot generate answer: {health_msg}")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            retriever = get_retriever(repo_name, mode=MODE_MAP[search_mode], top_k=top_k)
            llm = get_chat_model(provider=provider)
            try:
                with st.spinner(f"Searching `{repo_name}` via {search_mode}..."):
                    token_stream, retrieved_docs = stream_answer(user_query, retriever, llm=llm)
                full_answer = st.write_stream(token_stream)
            except Exception as exc:
                st.error(f"Generation error: {exc}")
                full_answer, retrieved_docs = f"Error during generation: {exc}", []

            valid, hallucinated = verify_all_citations(full_answer, retrieved_docs)
            render_citations_and_snippets(retrieved_docs, valid, hallucinated)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "docs": retrieved_docs,
                "valid_citations": valid,
                "hallucinated_citations": hallucinated,
            })
