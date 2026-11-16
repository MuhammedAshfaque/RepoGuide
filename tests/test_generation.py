"""
Unit tests for the generation pipeline (Phase 5).

Covers:
  - Prompt construction, few-shot presence, chunk truncation
  - Citation extraction regex (ranges and single lines)
  - Citation verification (valid, out-of-range, non-existent files)
  - AnswerResult grounding checks with a MockLLM provider
"""

from typing import Any, Iterator
import pytest

from src.generation.prompts import (
    SYSTEM_PROMPT,
    format_code_chunks,
    build_user_prompt,
    MAX_CHUNK_CHARS,
)
from src.generation.llm import LLMProvider
from src.generation.answerer import (
    extract_citations,
    verify_citation,
    verify_all_citations,
    ParsedCitation,
    AnswerResult,
    answer_question,
)


# ── Mock LLM Provider for Isolated Testing ───────────────────────────────────

class MockLLMProvider(LLMProvider):
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = ""
        self.last_system_prompt = ""

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return self.response_text

    def stream(self, prompt: str, system_prompt: str = "") -> Iterator[str]:
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        for word in self.response_text.split():
            yield word + " "

    def check_health(self) -> tuple[bool, str]:
        return True, "Mock is healthy"


# ═══════════════════════════════════════════════════════════════════════════
#  1. Prompt Construction Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPrompts:
    def test_system_prompt_contains_critical_rules(self):
        assert "I couldn't find this in the codebase" in SYSTEM_PROMPT
        assert "path/to/file.py:start_line-end_line" in SYSTEM_PROMPT
        assert "FEW-SHOT EXAMPLE" in SYSTEM_PROMPT

    def test_format_code_chunks_empty(self):
        result = format_code_chunks([])
        assert "No relevant code snippets" in result

    def test_format_code_chunks_respects_max_chunks(self):
        chunks = [
            {"file_path": f"file_{i}.py", "start_line": 1, "end_line": 10, "content": f"code_{i}"}
            for i in range(10)
        ]
        formatted = format_code_chunks(chunks, max_chunks=4)
        assert "Snippet [4]" in formatted
        assert "Snippet [5]" not in formatted

    def test_format_code_chunks_truncates_large_chunks(self):
        huge_code = "x = 1\n" * 1000
        chunks = [
            {"file_path": "big.py", "start_line": 1, "end_line": 1000, "content": huge_code}
        ]
        formatted = format_code_chunks(chunks, max_chunk_chars=200)
        assert "[truncated due to length]" in formatted
        assert len(formatted) < len(huge_code)

    def test_build_user_prompt(self):
        chunks = [
            {"file_path": "auth.py", "start_line": 5, "end_line": 15, "content": "def login(): pass"}
        ]
        prompt = build_user_prompt("Where is login?", chunks)
        assert "Question: Where is login?" in prompt
        assert "auth.py:5-15" in prompt
        assert "def login(): pass" in prompt


# ═══════════════════════════════════════════════════════════════════════════
#  2. Citation Extraction & Verification Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCitationVerification:
    @pytest.fixture
    def retrieved_chunks(self) -> list[dict[str, Any]]:
        return [
            {
                "file_path": "conduit/user/views.py",
                "start_line": 30,
                "end_line": 40,
                "content": "def login_user(): pass",
            },
            {
                "file_path": "conduit/user/models.py",
                "start_line": 33,
                "end_line": 35,
                "content": "def check_password(): pass",
            },
        ]

    def test_extract_citations_various_formats(self):
        text = (
            "Authentication is in `conduit/user/views.py:30-40` and password checking "
            "is at conduit\\user\\models.py:33. Also see `main.py:100-120`."
        )
        citations = extract_citations(text)
        assert len(citations) == 3

        assert citations[0].file_path == "conduit/user/views.py"
        assert citations[0].start_line == 30
        assert citations[0].end_line == 40

        assert citations[1].file_path == "conduit/user/models.py"
        assert citations[1].start_line == 33
        assert citations[1].end_line == 33

    def test_verify_citation_valid(self, retrieved_chunks):
        cite = ParsedCitation("views.py:30-40", "conduit/user/views.py", 30, 40)
        assert verify_citation(cite, retrieved_chunks) is True

    def test_verify_citation_within_boundary_margin(self, retrieved_chunks):
        # Line 28 is within the 5-line margin of start_line=30
        cite = ParsedCitation("views.py:28-38", "conduit/user/views.py", 28, 38)
        assert verify_citation(cite, retrieved_chunks) is True

    def test_verify_citation_wrong_file(self, retrieved_chunks):
        cite = ParsedCitation("fake.py:30-40", "conduit/fake.py", 30, 40)
        assert verify_citation(cite, retrieved_chunks) is False

    def test_verify_citation_out_of_line_range(self, retrieved_chunks):
        cite = ParsedCitation("views.py:80-95", "conduit/user/views.py", 80, 95)
        assert verify_citation(cite, retrieved_chunks) is False

    def test_verify_all_citations(self, retrieved_chunks):
        answer = (
            "User login is at `conduit/user/views.py:30-40`. "
            "Database setup is in `conduit/database.py:1-50`."
        )
        valid, hallucinated = verify_all_citations(answer, retrieved_chunks)
        assert "conduit/user/views.py:30-40" in valid
        assert "conduit/database.py:1-50" in hallucinated


# ═══════════════════════════════════════════════════════════════════════════
#  3. Answer Pipeline Integration Tests with MockLLM
# ═══════════════════════════════════════════════════════════════════════════

class TestAnswerPipelineWithMock:
    def test_answer_question_with_grounded_mock(self, monkeypatch):
        mock_chunks = [
            {
                "file_path": "conduit/user/views.py",
                "start_line": 30,
                "end_line": 40,
                "content": "def login_user(): pass",
                "symbol_name": "login_user",
                "chunk_type": "function",
                "score": 0.03,
                "rank": 1,
            }
        ]
        # Mock hybrid_search so no vector/BM25 models are needed
        monkeypatch.setattr(
            "src.generation.answerer.hybrid_search",
            lambda query, repo_name, top_k_final: mock_chunks,
        )

        mock_llm = MockLLMProvider("Handled at `conduit/user/views.py:30-40`.")
        result = answer_question("Where is login?", "test_repo", llm=mock_llm)

        assert isinstance(result, AnswerResult)
        assert "conduit/user/views.py:30-40" in result.valid_citations
        assert len(result.hallucinated_citations) == 0
        assert result.is_grounded is True

    def test_answer_question_detects_hallucination(self, monkeypatch):
        mock_chunks = [
            {
                "file_path": "conduit/user/views.py",
                "start_line": 30,
                "end_line": 40,
                "content": "def login_user(): pass",
                "symbol_name": "login_user",
                "chunk_type": "function",
                "score": 0.03,
                "rank": 1,
            }
        ]
        monkeypatch.setattr(
            "src.generation.answerer.hybrid_search",
            lambda query, repo_name, top_k_final: mock_chunks,
        )

        # LLM cites an invented file
        mock_llm = MockLLMProvider("See `conduit/fake_auth.py:10-20`.")
        result = answer_question("Where is login?", "test_repo", llm=mock_llm)

        assert "conduit/fake_auth.py:10-20" in result.hallucinated_citations
        assert result.is_grounded is False

