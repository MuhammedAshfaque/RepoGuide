"""
End-to-end question answering pipeline with citation verification for CodeCompass.

Pipeline:
  1. Retrieve top-k chunks using hybrid search (Vector + BM25 via RRF).
  2. Format code snippets into the prompt with few-shot citation guidance.
  3. Query the LLM provider (Ollama by default).
  4. Perform post-generation verification: regex-extract all cited file paths
     and line numbers, and check whether they actually exist in the retrieved chunks.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.config import TOP_K_FINAL
from src.retrieval.hybrid import hybrid_search
from src.generation.llm import LLMProvider, get_llm_provider
from src.generation.prompts import SYSTEM_PROMPT, build_user_prompt


# ── Citation Extraction & Verification ───────────────────────────────────────

# Matches patterns like:
#   `conduit/user/views.py:30-40`
#   conduit\user\views.py:30
#   src/auth/service.py:12-24
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
    """
    Extract all file and line citations from model output.

    Handles both range citations (`file.py:10-25`) and single-line citations (`file.py:10`).
    """
    citations: list[ParsedCitation] = []
    for match in _CITATION_REGEX.finditer(text):
        raw = match.group(0).strip("`")
        fpath = match.group(1).replace("\\", "/").strip()
        start = int(match.group(2))
        end = int(match.group(3)) if match.group(3) else start

        citations.append(ParsedCitation(
            raw=raw,
            file_path=fpath,
            start_line=start,
            end_line=end,
        ))
    return citations


def verify_citation(citation: ParsedCitation, retrieved_chunks: list[dict[str, Any]]) -> bool:
    """
    Verify whether a cited file and line range actually exists in the retrieved chunks.

    A citation is valid if:
      1. Its file path matches a chunk's file path (case-insensitive, normalized slashes).
      2. Its line numbers overlap with or fall within the chunk's line range
         (with a generous 5-line margin for header/decorator offsets).
    """
    norm_cite_path = citation.file_path.lower().replace("\\", "/")

    for chunk in retrieved_chunks:
        chunk_path = chunk.get("file_path", "").lower().replace("\\", "/")

        # Check file match (exact match or suffix match if relative)
        if norm_cite_path == chunk_path or chunk_path.endswith(norm_cite_path):
            c_start = int(chunk.get("start_line", 0))
            c_end = int(chunk.get("end_line", 0))

            # Allow a small 5-line boundary buffer
            margin = 5
            valid_range_start = max(1, c_start - margin)
            valid_range_end = c_end + margin

            # Check for range overlap: max(start1, start2) <= min(end1, end2)
            overlap_start = max(citation.start_line, valid_range_start)
            overlap_end = min(citation.end_line, valid_range_end)

            if overlap_start <= overlap_end:
                return True

    return False


def verify_all_citations(
    text: str,
    retrieved_chunks: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """
    Extract and verify all citations in the generated text.

    Returns:
        (valid_citations, hallucinated_citations)
    """
    parsed = extract_citations(text)
    valid: list[str] = []
    hallucinated: list[str] = []

    for cite in parsed:
        if verify_citation(cite, retrieved_chunks):
            if cite.raw not in valid:
                valid.append(cite.raw)
        else:
            if cite.raw not in hallucinated:
                hallucinated.append(cite.raw)

    return valid, hallucinated


# ── Answer Result Container ──────────────────────────────────────────────────

@dataclass
class AnswerResult:
    """Encapsulates the complete Q&A response and citation verification audit."""
    query: str
    answer: str
    retrieved_chunks: list[dict[str, Any]] = field(default_factory=list)
    valid_citations: list[str] = field(default_factory=list)
    hallucinated_citations: list[str] = field(default_factory=list)
    is_grounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "retrieved_chunks_count": len(self.retrieved_chunks),
            "valid_citations": self.valid_citations,
            "hallucinated_citations": self.hallucinated_citations,
            "is_grounded": self.is_grounded,
        }


# ── Main Answerer Pipeline ───────────────────────────────────────────────────

def answer_question(
    query: str,
    repo_name: str,
    top_k: int = TOP_K_FINAL,
    llm: LLMProvider | None = None,
) -> AnswerResult:
    """
    Retrieve code snippets, generate an answer with citations, and verify them.

    Args:
        query: User question.
        repo_name: Repository name.
        top_k: Number of chunks to feed into prompt (default: 4).
        llm: LLM provider (defaults to configured provider).

    Returns:
        AnswerResult containing answer, chunks, and verified citations.
    """
    provider = llm or get_llm_provider()

    # Step 1: Hybrid retrieval
    chunks = hybrid_search(query, repo_name, top_k_final=top_k)

    if not chunks:
        return AnswerResult(
            query=query,
            answer="I couldn't find this in the codebase. No relevant code chunks were retrieved.",
            retrieved_chunks=[],
            valid_citations=[],
            hallucinated_citations=[],
            is_grounded=True,
        )

    # Step 2: Build prompt with few-shot citation instructions
    user_prompt = build_user_prompt(query, chunks)

    # Step 3: Generate response
    raw_answer = provider.generate(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)

    # Step 4: Verify citations
    valid, hallucinated = verify_all_citations(raw_answer, chunks)
    is_grounded = len(hallucinated) == 0

    return AnswerResult(
        query=query,
        answer=raw_answer.strip(),
        retrieved_chunks=chunks,
        valid_citations=valid,
        hallucinated_citations=hallucinated,
        is_grounded=is_grounded,
    )


def stream_answer_question(
    query: str,
    repo_name: str,
    top_k: int = TOP_K_FINAL,
    llm: LLMProvider | None = None,
) -> tuple[Iterator[str], list[dict[str, Any]]]:
    """
    Streaming generator for UI integration (Phase 6).

    Returns:
        (stream_iterator, retrieved_chunks)
    """
    provider = llm or get_llm_provider()
    chunks = hybrid_search(query, repo_name, top_k_final=top_k)

    if not chunks:
        def _empty_gen():
            yield "I couldn't find this in the codebase."
        return _empty_gen(), []

    user_prompt = build_user_prompt(query, chunks)
    stream_iter = provider.stream(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)

    return stream_iter, chunks


# ── CLI for testing ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.generation.answerer <repo_name> <query>")
        print('Example: python -m src.generation.answerer flask-realworld-example-app "Where is user authentication handled?"')
        sys.exit(1)

    repo_name = sys.argv[1]
    query = " ".join(sys.argv[2:])

    print(f"\n{'='*70}")
    print(f" 🤖 CodeCompass Question Answerer")
    print(f" ❓ Question: \"{query}\"")
    print(f" 📦 Repo:     {repo_name}")
    print(f"{'='*70}\n")

    result = answer_question(query, repo_name)

    print("--- 📝 Answer ---")
    print(result.answer)
    print("\n--- 🔍 Citation Audit ---")
    print(f"Retrieved Chunks: {len(result.retrieved_chunks)}")
    for i, c in enumerate(result.retrieved_chunks, 1):
        print(f"  [{i}] {c['file_path']}:{c['start_line']}-{c['end_line']} ({c.get('symbol_name')})")

    if result.valid_citations:
        print(f"✅ Verified Citations: {result.valid_citations}")
    if result.hallucinated_citations:
        print(f"⚠️  Hallucinated / Unverified Citations: {result.hallucinated_citations}")
    if not result.valid_citations and not result.hallucinated_citations:
        print("ℹ️  No specific line citations present.")

    print(f"\nGrounded in Context: {'YES ✅' if result.is_grounded else 'NO ⚠️'}")
    print(f"{'='*70}\n")

