"""
Unified LCEL generation chain and citation verification for CodeCompass.

Combines:
  1. Chat model factory — ChatOllama (default) or ChatGoogleGenerativeAI via LLM_PROVIDER
  2. ChatPromptTemplate — strict grounding rules, refusal phrase, and few-shot examples
  3. Chunk formatter — formats at most 4 LangChain Documents with length safeguard
  4. LCEL chain — retriever -> format_docs -> prompt -> LLM -> StrOutputParser (with streaming)
  5. Citation verification — custom regex post-check against retrieved Documents

Memory notes (8 GB RAM):
  - ChatOllama defaults to qwen2.5-coder:3b with num_ctx=4096 and temperature=0.1.
  - At most 4 chunks (each capped at 2500 chars) are sent to the LLM.
"""

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


# ── Prompt Template (Strict Grounding + Few-Shot Examples) ───────────────────
# Note: literal braces inside code examples are escaped as {{ }} for ChatPromptTemplate.

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

HUMAN_TEMPLATE = """Question: {question}

Retrieved Code Snippets:
{context}

Answer:"""


def get_prompt_template() -> ChatPromptTemplate:
    """Build the ChatPromptTemplate with system grounding rules and few-shot examples."""
    return ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_TEMPLATE),
    ])


# ── Chat Model Factory & Health Check ────────────────────────────────────────

def get_chat_model(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseChatModel:
    """
    Create a LangChain chat model for the configured provider.

    Args:
        provider: 'ollama' (default) or 'gemini'. Falls back to LLM_PROVIDER from .env.
        model: Optional model override. Falls back to LLM_MODEL or GEMINI_MODEL.
    """
    chosen = (provider or LLM_PROVIDER).lower().strip()

    if chosen == "ollama":
        return ChatOllama(
            model=model or LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=LLM_TEMPERATURE,
            num_ctx=LLM_NUM_CTX,
        )

    if chosen == "gemini":
        if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
            raise ValueError(
                "GEMINI_API_KEY is not set in .env. "
                "Add GEMINI_API_KEY=... to your .env file to use Gemini."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=LLM_TEMPERATURE,
        )

    raise ValueError(
        f"Unsupported LLM provider '{chosen}'. Supported providers: 'ollama', 'gemini'."
    )


def check_llm_health(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Check whether the selected LLM backend is reachable and configured.

    Returns:
        (is_healthy, status_message)
    """
    chosen = (provider or LLM_PROVIDER).lower().strip()

    if chosen == "gemini":
        target_model = model or GEMINI_MODEL
        if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
            return (
                False,
                "GEMINI_API_KEY is not configured in .env. "
                "Add GEMINI_API_KEY=... to use Gemini.",
            )
        return (True, f"Gemini API configured ({target_model})")

    if chosen == "ollama":
        target_model = model or LLM_MODEL
        try:
            import ollama

            client = ollama.Client(host=OLLAMA_BASE_URL)
            models_resp = client.list()
        except Exception as exc:
            return (
                False,
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
                f"Make sure Ollama is running. ({exc})",
            )

        models_list = getattr(models_resp, "models", None)
        if models_list is None and isinstance(models_resp, dict):
            models_list = models_resp.get("models", [])

        available = []
        for m in models_list or []:
            name = getattr(m, "model", None) or getattr(m, "name", None)
            if not name and isinstance(m, dict):
                name = m.get("model") or m.get("name")
            if name:
                available.append(name.lower())

        target_lower = target_model.lower()
        found = any(
            target_lower == name
            or name.startswith(f"{target_lower}:")
            or target_lower.startswith(f"{name}:")
            for name in available
        )
        if not found:
            return (
                False,
                f"Model '{target_model}' is not pulled in Ollama. "
                f"Run: `ollama pull {target_model}`",
            )
        return (True, f"Ollama ready ({target_model})")

    return (False, f"Unsupported LLM provider: '{chosen}'")


# ── Context Formatting ───────────────────────────────────────────────────────

def format_docs(
    docs: list[Document],
    max_chunks: int = TOP_K_FINAL,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> str:
    """
    Format up to `max_chunks` LangChain Documents into labeled code snippets.

    Enforces both chunk count (default 4) and per-chunk character limit (2500)
    to stay within the 4096-token context window on 8 GB RAM machines.
    """
    if not docs:
        return "(No relevant code snippets found.)"

    blocks: list[str] = []
    for idx, doc in enumerate(docs[:max_chunks], 1):
        m = doc.metadata
        file_path = m.get("file_path", "unknown")
        start = m.get("start_line", 0)
        end = m.get("end_line", 0)
        symbol = m.get("symbol_name") or "(block)"
        chunk_type = m.get("chunk_type") or "code"
        parent = f" in {m['parent_class']}" if m.get("parent_class") else ""

        header = f"--- Snippet [{idx}]: {file_path}:{start}-{end} ({chunk_type}: {symbol}{parent}) ---"
        content = strip_metadata_header(doc.page_content).strip()
        if len(content) > max_chunk_chars:
            content = content[:max_chunk_chars] + "\n... [truncated due to length]"

        blocks.append(f"{header}\n{content}")

    return "\n\n".join(blocks)


# ── LCEL Chain ───────────────────────────────────────────────────────────────

def build_chain(
    retriever: Runnable[str, list[Document]],
    llm: Optional[BaseChatModel] = None,
) -> Runnable[str, str]:
    """
    Build the LCEL RAG chain:
      question -> {context: retriever | format_docs, question: passthrough}
               -> ChatPromptTemplate -> ChatModel -> StrOutputParser

    Supports both `.invoke(question)` and `.stream(question)`.
    """
    model = llm or get_chat_model()
    prompt = get_prompt_template()

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )


def stream_answer(
    question: str,
    retriever: Runnable[str, list[Document]],
    llm: Optional[BaseChatModel] = None,
) -> tuple[Iterator[str], list[Document]]:
    """
    Retrieve documents once and stream the LCEL generation response.

    Returns:
        (token_iterator, retrieved_docs) so callers (e.g. Streamlit) can stream
        tokens, display source snippets, and run verify_all_citations() after.
    """
    docs = retriever.invoke(question)[:TOP_K_FINAL]
    if not docs:
        def _refusal_stream() -> Iterator[str]:
            yield f"{REFUSAL_MESSAGE} No relevant code chunks were retrieved."
        return _refusal_stream(), []

    model = llm or get_chat_model()
    gen_chain = get_prompt_template() | model | StrOutputParser()
    stream_iter = gen_chain.stream({
        "context": format_docs(docs),
        "question": question.strip(),
    })
    return stream_iter, docs


# ── Citation Extraction & Verification (Custom Post-Check) ───────────────────

_CITATION_REGEX = re.compile(
    r"`?([a-zA-Z0-9_\-./\\]+\.[a-zA-Z0-9]+):(\d+)(?:-(\d+))?`?"
)


@dataclass
class ParsedCitation:
    raw: str
    file_path: str
    start_line: int
    end_line: int


def extract_citations(text: str) -> list[ParsedCitation]:
    """Extract all `file.py:start-end` and `file.py:line` citations from text."""
    citations: list[ParsedCitation] = []
    for match in _CITATION_REGEX.finditer(text):
        raw = match.group(0).strip("`")
        fpath = match.group(1).replace("\\", "/").strip()
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start
        citations.append(ParsedCitation(raw=raw, file_path=fpath, start_line=start, end_line=end))
    return citations


def verify_citation(citation: ParsedCitation, retrieved_docs: list[Document]) -> bool:
    """
    Check whether a cited file and line range matches one of the retrieved Documents.

    Allows a +/- 5-line buffer around the chunk boundaries for decorator/header offsets.
    """
    norm_cite_path = citation.file_path.lower().replace("\\", "/")

    for doc in retrieved_docs:
        m = doc.metadata
        doc_path = str(m.get("file_path", "")).lower().replace("\\", "/")
        if norm_cite_path == doc_path or doc_path.endswith(norm_cite_path):
            c_start = int(m.get("start_line", 0))
            c_end = int(m.get("end_line", 0))
            margin = 5
            if max(citation.start_line, max(1, c_start - margin)) <= min(citation.end_line, c_end + margin):
                return True

    return False


def verify_all_citations(
    text: str,
    retrieved_docs: list[Document],
) -> tuple[list[str], list[str]]:
    """
    Audit all citations in generated text against retrieved Documents.

    Returns:
        (valid_citations, hallucinated_citations)
    """
    valid: list[str] = []
    hallucinated: list[str] = []

    for cite in extract_citations(text):
        if verify_citation(cite, retrieved_docs):
            if cite.raw not in valid:
                valid.append(cite.raw)
        else:
            if cite.raw not in hallucinated:
                hallucinated.append(cite.raw)

    return valid, hallucinated


# ── CLI for quick testing ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from src.retrieval.retrieval import get_retriever

    parser = argparse.ArgumentParser(description="Ask a question using the LangChain LCEL chain.")
    parser.add_argument("repo_name", help="Indexed repository name")
    parser.add_argument("question", help="Question about the codebase")
    parser.add_argument("--mode", default="hybrid", choices=["vector", "bm25", "hybrid", "hybrid_rerank"])
    parser.add_argument("--provider", default=None, choices=["ollama", "gemini"])
    args = parser.parse_args()

    retriever = get_retriever(args.repo_name, mode=args.mode, top_k=TOP_K_FINAL)
    llm = get_chat_model(provider=args.provider)

    print(f"\n🤖 Question: \"{args.question}\" | Repo: {args.repo_name} | Mode: {args.mode}")
    print("=" * 70)

    token_stream, docs = stream_answer(args.question, retriever, llm=llm)
    chunks_out: list[str] = []
    for token in token_stream:
        print(token, end="", flush=True)
        chunks_out.append(token)
    print("\n")

    full_answer = "".join(chunks_out)
    valid, hallucinated = verify_all_citations(full_answer, docs)
    print(f"Retrieved: {len(docs)} chunks | Verified: {valid} | Unverified: {hallucinated}")
    print("=" * 70)
