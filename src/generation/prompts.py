"""
Prompt templates with few-shot citation examples for CodeCompass.

Implements:
  1. Strict system instructions:
     - Answer ONLY from retrieved code snippets.
     - Strict refusal phrase if context is insufficient:
       "I couldn't find this in the codebase."
     - Mandatory citation syntax: `path/to/file.py:start_line-end_line`.
  2. Concrete few-shot examples (positive & negative), which is critical
     for smaller 3B models like `qwen2.5-coder:3b` to maintain formatting discipline.
  3. Chunk context formatting with truncation safeguards.
"""

from typing import Any

# Maximum characters allowed per individual chunk to prevent context blowout
MAX_CHUNK_CHARS = 2500
MAX_CHUNKS_DEFAULT = 4


# ── System Prompt with Few-Shot Citation Examples ────────────────────────────

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
    return {}

Answer:
I couldn't find this in the codebase. The provided code snippets do not mention payment processing or Stripe integration.
"""


# ── Context Formatting Helpers ───────────────────────────────────────────────

def format_code_chunks(
    chunks: list[dict[str, Any]],
    max_chunks: int = MAX_CHUNKS_DEFAULT,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> str:
    """
    Format retrieved code chunks into a clear, labeled context block.

    Args:
        chunks: List of chunk dictionaries from retrieval.
        max_chunks: Maximum number of chunks to include (default: 4).
        max_chunk_chars: Maximum characters per chunk body.

    Returns:
        Formatted context string.
    """
    if not chunks:
        return "(No relevant code snippets found.)"

    selected_chunks = chunks[:max_chunks]
    formatted_blocks: list[str] = []

    for idx, chunk in enumerate(selected_chunks, 1):
        file_path = chunk.get("file_path", "unknown")
        start = chunk.get("start_line", 0)
        end = chunk.get("end_line", 0)
        symbol = chunk.get("symbol_name") or "(block)"
        chunk_type = chunk.get("chunk_type") or "code"
        parent = f" in {chunk['parent_class']}" if chunk.get("parent_class") else ""

        header = f"--- Snippet [{idx}]: {file_path}:{start}-{end} ({chunk_type}: {symbol}{parent}) ---"

        raw_content = chunk.get("content", "").strip()
        if len(raw_content) > max_chunk_chars:
            content = raw_content[:max_chunk_chars] + "\n... [truncated due to length]"
        else:
            content = raw_content

        formatted_blocks.append(f"{header}\n{content}")

    return "\n\n".join(formatted_blocks)


def build_user_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    """
    Construct the complete user prompt from the query and retrieved chunks.

    Args:
        query: User question.
        chunks: Retrieved code chunks.

    Returns:
        Prompt string ready to be passed to LLM.
    """
    context_str = format_code_chunks(chunks)

    prompt = (
        f"Question: {query.strip()}\n\n"
        f"Retrieved Code Snippets:\n"
        f"{context_str}\n\n"
        f"Answer:"
    )
    return prompt

