"""LCEL generation chain (ChatOllama / Gemini), strict citation prompt, and regex citation audit."""

import re
from dataclasses import dataclass
from typing import Iterator, Optional

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    LLM_MODEL,
    LLM_NUM_CTX,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    TOP_K_FINAL,
)
from src.ingestion.indexer import strip_metadata_header

MAX_CHUNK_CHARS = 2500
REFUSAL_MESSAGE = "I couldn't find this in the codebase."

# Literal braces inside few-shot code examples are escaped as {{ }} for ChatPromptTemplate.
SYSTEM_PROMPT = """You are CodeCompass, a precise codebase onboarding assistant.
Your job is to answer the user's question about a codebase using ONLY the provided code snippets.

STRICT RULES YOU MUST FOLLOW:
1. GROUNDED IN CONTEXT: Use ONLY the provided code snippets. Do not guess, speculate, or invent facts outside the provided snippets.
2. CITATION REQUIREMENT: Every claim, function explanation, or architectural statement MUST include a citation with the exact relative file path and line numbers in this format: `path/to/file.py:start_line-end_line` (or `path/to/file.py:line`).
3. INSUFFICIENT CONTEXT: If the provided code snippets do not contain enough information to answer the question, you MUST say:
   "I couldn't find this in the codebase."
   Do NOT provide generic programming advice when the answer is not in the code.
4. CODE EXCERPTS: When citing code, reference the symbol names, file paths, and exact lines. Keep answers concise, technical, and developer-friendly.

--- FEW-SHOT EXAMPLE 1 (Sufficient Context) ---
Question: Where is user authentication handled?

Context:
--- Snippet [1]: conduit/user/views.py:30-40 (function: login_user) ---
@blueprint.route('/api/users/login', methods=('POST',))
@jwt_optional
@use_kwargs(user_schema)
def login_user(user):
    return user

--- Snippet [2]: conduit/user/models.py:33-35 (method: check_password) ---
def check_password(self, value):
    return bcrypt.check_password_hash(self.password, value)

Answer:
User authentication is handled in `conduit/user/views.py:30-40` through the `login_user` endpoint (`POST /api/users/login`). Password verification is implemented in the `User` model at `conduit/user/models.py:33-35` via the `check_password` method using bcrypt.

--- FEW-SHOT EXAMPLE 2 (Insufficient Context) ---
Question: How does the application process Stripe credit card payments?

Context:
--- Snippet [1]: conduit/articles/views.py:39-54 (function: make_article) ---
@blueprint.route('/api/articles', methods=('POST',))
def make_article():
    return {{}}

Answer:
I couldn't find this in the codebase. The provided code snippets do not mention payment processing or Stripe integration.
"""

HUMAN_TEMPLATE = "Question: {question}\n\nRetrieved Code Snippets:\n{context}\n\nAnswer:"


def get_prompt_template() -> ChatPromptTemplate:
    """Return the grounded ChatPromptTemplate with few-shot citation examples."""
    return ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_TEMPLATE)])


def get_chat_model(provider: Optional[str] = None, model: Optional[str] = None) -> BaseChatModel:
    """Instantiate ChatOllama (default) or ChatGoogleGenerativeAI based on LLM_PROVIDER."""
    chosen = (provider or LLM_PROVIDER).lower().strip()
    if chosen == "ollama":
        return ChatOllama(
            model=model or LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_ctx=LLM_NUM_CTX,
        )
    if chosen == "gemini":
        if not GEMINI_API_KEY.strip():
            raise ValueError("GEMINI_API_KEY is not set in .env.")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=LLM_TEMPERATURE,
        )
    raise ValueError(f"Unsupported LLM provider '{chosen}'. Use 'ollama' or 'gemini'.")


def check_llm_health(provider: Optional[str] = None, model: Optional[str] = None) -> tuple[bool, str]:
    """Return (is_healthy, message) for the selected LLM provider."""
    chosen = (provider or LLM_PROVIDER).lower().strip()
    if chosen == "gemini":
        if not GEMINI_API_KEY.strip():
            return False, "GEMINI_API_KEY is not configured in .env."
        return True, f"Gemini API configured ({model or GEMINI_MODEL})"

    if chosen != "ollama":
        return False, f"Unsupported LLM provider: '{chosen}'"

    target = (model or LLM_MODEL).lower()
    try:
        import ollama
        resp = ollama.Client(host=OLLAMA_BASE_URL).list()
        models = getattr(resp, "models", None) or (resp.get("models", []) if isinstance(resp, dict) else [])
    except Exception as exc:
        return False, f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. ({exc})"

    names = [
        (getattr(m, "model", None) or getattr(m, "name", None) or (m.get("model") if isinstance(m, dict) else "") or "").lower()
        for m in models
    ]
    if any(target == n or n.startswith(f"{target}:") or target.startswith(f"{n}:") for n in names if n):
        return True, f"Ollama ready ({model or LLM_MODEL})"
    return False, f"Model '{model or LLM_MODEL}' not found in Ollama. Run: `ollama pull {model or LLM_MODEL}`"


def format_docs(docs: list[Document], max_chunks: int = TOP_K_FINAL, max_chunk_chars: int = MAX_CHUNK_CHARS) -> str:
    """Format up to max_chunks Documents into labeled code blocks within the context budget."""
    if not docs:
        return "(No relevant code snippets found.)"

    blocks: list[str] = []
    for idx, doc in enumerate(docs[:max_chunks], 1):
        m = doc.metadata
        parent = f" in {m['parent_class']}" if m.get("parent_class") else ""
        header = (
            f"--- Snippet [{idx}]: {m.get('file_path', 'unknown')}:{m.get('start_line', 0)}-{m.get('end_line', 0)} "
            f"({m.get('chunk_type') or 'code'}: {m.get('symbol_name') or '(block)'}{parent}) ---"
        )
        content = strip_metadata_header(doc.page_content).strip()
        if len(content) > max_chunk_chars:
            content = content[:max_chunk_chars] + "\n... [truncated due to length]"
        blocks.append(f"{header}\n{content}")
    return "\n\n".join(blocks)


def build_chain(retriever: Runnable[str, list[Document]], llm: Optional[BaseChatModel] = None) -> Runnable[str, str]:
    """Build the LCEL RAG pipeline: retriever -> format_docs -> prompt -> LLM -> StrOutputParser."""
    return (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | get_prompt_template()
        | (llm or get_chat_model())
        | StrOutputParser()
    )


def stream_answer(
    question: str,
    retriever: Runnable[str, list[Document]],
    llm: Optional[BaseChatModel] = None,
) -> tuple[Iterator[str], list[Document]]:
    """Retrieve documents once and stream the LCEL response alongside the retrieved Documents."""
    docs = retriever.invoke(question)[:TOP_K_FINAL]
    if not docs:
        return iter([f"{REFUSAL_MESSAGE} No relevant code chunks were retrieved."]), []

    gen_chain = get_prompt_template() | (llm or get_chat_model()) | StrOutputParser()
    return gen_chain.stream({"context": format_docs(docs), "question": question.strip()}), docs


# ── Custom Citation Verification ─────────────────────────────────────────────

_CITATION_REGEX = re.compile(r"`?([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+):(\d+)(?:-(\d+))?`?")


@dataclass
class ParsedCitation:
    raw: str
    file_path: str
    start_line: int
    end_line: int


def extract_citations(text: str) -> list[ParsedCitation]:
    """Extract all `file.py:start-end` and `file.py:line` citations from generated text."""
    return [
        ParsedCitation(
            raw=m.group(0).strip("`"),
            file_path=m.group(1).replace("\\", "/").strip(),
            start_line=int(m.group(2)),
            end_line=int(m.group(3) or m.group(2)),
        )
        for m in _CITATION_REGEX.finditer(text)
    ]


def verify_citation(citation: ParsedCitation, retrieved_docs: list[Document]) -> bool:
    """Return True if a citation matches a retrieved Document's file and line range (+/- 5 lines)."""
    cite_path = citation.file_path.lower().replace("\\", "/")
    for doc in retrieved_docs:
        m = doc.metadata
        doc_path = str(m.get("file_path", "")).lower().replace("\\", "/")
        if cite_path == doc_path or doc_path.endswith(cite_path):
            c_start, c_end = int(m.get("start_line", 0)), int(m.get("end_line", 0))
            if max(citation.start_line, max(1, c_start - 5)) <= min(citation.end_line, c_end + 5):
                return True
    return False


def verify_all_citations(text: str, retrieved_docs: list[Document]) -> tuple[list[str], list[str]]:
    """Split extracted citations into (valid_citations, hallucinated_citations)."""
    valid: list[str] = []
    hallucinated: list[str] = []
    for cite in extract_citations(text):
        bucket = valid if verify_citation(cite, retrieved_docs) else hallucinated
        if cite.raw not in bucket:
            bucket.append(cite.raw)
    return valid, hallucinated
