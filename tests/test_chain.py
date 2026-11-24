"""
Tests for unified LCEL generation chain and citation verification (src/generation/chain.py).

Covers:
  - ChatPromptTemplate rules, few-shot examples, and escaping
  - format_docs() max-chunk cap and character truncation
  - get_chat_model() provider switching and error handling
  - stream_answer() streaming with mock retriever and mock LLM
  - stream_answer() refusal when no documents are retrieved
  - Citation extraction and post-generation verification against Documents
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.generation import chain as chain_mod
from src.generation.chain import (
    ParsedCitation,
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    extract_citations,
    format_docs,
    get_chat_model,
    get_prompt_template,
    stream_answer,
    verify_all_citations,
    verify_citation,
)


def _make_doc(file_path: str, start: int, end: int, content: str, symbol: str = "fn") -> Document:
    return Document(
        page_content=content,
        metadata={
            "file_path": file_path,
            "start_line": start,
            "end_line": end,
            "symbol_name": symbol,
            "chunk_type": "function",
        },
    )


# ── 1. Prompt & Formatting Tests ─────────────────────────────────────────────

class TestPromptAndFormatting:
    def test_system_prompt_has_rules_and_few_shot(self):
        assert REFUSAL_MESSAGE in SYSTEM_PROMPT
        assert "path/to/file.py:start_line-end_line" in SYSTEM_PROMPT
        assert "FEW-SHOT EXAMPLE 1" in SYSTEM_PROMPT
        assert "FEW-SHOT EXAMPLE 2" in SYSTEM_PROMPT

    def test_prompt_template_renders_without_brace_error(self):
        prompt = get_prompt_template()
        messages = prompt.format_messages(
            question="Where is login?",
            context="--- Snippet [1]: auth.py:1-5 ---\ndef login(): pass",
        )
        assert len(messages) == 2
        assert "Where is login?" in messages[1].content
        assert "return {}" in messages[0].content

    def test_format_docs_empty(self):
        assert "No relevant code snippets" in format_docs([])

    def test_format_docs_respects_max_chunks(self):
        docs = [_make_doc(f"f{i}.py", 1, 10, f"code_{i}") for i in range(8)]
        out = format_docs(docs, max_chunks=4)
        assert "Snippet [4]" in out
        assert "Snippet [5]" not in out

    def test_format_docs_truncates_long_chunk(self):
        big = "x = 1\n" * 800
        out = format_docs([_make_doc("big.py", 1, 800, big)], max_chunk_chars=200)
        assert "[truncated due to length]" in out
        assert len(out) < len(big)

    def test_metadata_header_enrichment_and_stripping(self):
        from src.ingestion.indexer import add_metadata_header, strip_metadata_header

        raw = Document(
            page_content="def check_password(self, val):\n    return True",
            metadata={
                "file_path": "conduit\\user\\models.py",
                "start_line": 33,
                "end_line": 35,
                "symbol_name": "check_password",
                "chunk_type": "method",
                "parent_class": "User",
            },
        )
        enriched = add_metadata_header(raw)
        assert enriched.page_content.startswith("# method User.check_password | File: conduit/user/models.py\n")
        assert enriched.metadata["start_line"] == 33
        assert enriched.metadata["end_line"] == 35
        assert strip_metadata_header(enriched.page_content) == raw.page_content
        # format_docs strips the synthetic header line so it's not duplicated
        formatted = format_docs([enriched])
        assert "# method User.check_password | File:" not in formatted
        assert "def check_password(self, val):" in formatted


# ── 2. Model Factory Tests ───────────────────────────────────────────────────

class TestChatModelFactory:
    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            get_chat_model(provider="unknown_xyz")

    def test_gemini_without_key_raises(self, monkeypatch):
        monkeypatch.setattr(chain_mod, "GEMINI_API_KEY", "")
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            get_chat_model(provider="gemini")


# ── 3. LCEL Chain & Streaming Tests ──────────────────────────────────────────

class TestLCELChain:
    def test_stream_answer_streams_and_returns_docs(self):
        docs = [_make_doc("conduit/user/views.py", 30, 40, "def login_user(): pass", "login_user")]
        fake_retriever = RunnableLambda(lambda q: docs)
        fake_llm = RunnableLambda(
            lambda prompt_val: AIMessage(content="See `conduit/user/views.py:30-40`.")
        )

        token_stream, returned_docs = stream_answer("Where is login?", fake_retriever, llm=fake_llm)
        full_text = "".join(token_stream)

        assert len(returned_docs) == 1
        assert "conduit/user/views.py:30-40" in full_text

    def test_stream_answer_refuses_when_no_docs(self):
        empty_retriever = RunnableLambda(lambda q: [])
        token_stream, returned_docs = stream_answer("Stripe payments?", empty_retriever)
        full_text = "".join(token_stream)

        assert returned_docs == []
        assert REFUSAL_MESSAGE in full_text


# ── 4. Citation Verification Tests ───────────────────────────────────────────

class TestCitationVerification:
    @pytest.fixture
    def retrieved_docs(self) -> list[Document]:
        return [
            _make_doc("conduit/user/views.py", 30, 40, "def login_user(): pass"),
            _make_doc("conduit/user/models.py", 33, 35, "def check_password(): pass"),
        ]

    def test_extract_citations_ranges_and_single_lines(self):
        text = (
            "Auth in `conduit/user/views.py:30-40`, password at "
            "conduit\\user\\models.py:33, also `main.py:100-120`."
        )
        cites = extract_citations(text)
        assert len(cites) == 3
        assert (cites[0].file_path, cites[0].start_line, cites[0].end_line) == ("conduit/user/views.py", 30, 40)
        assert (cites[1].file_path, cites[1].start_line, cites[1].end_line) == ("conduit/user/models.py", 33, 33)

    def test_verify_citation_valid_and_within_margin(self, retrieved_docs):
        assert verify_citation(ParsedCitation("views.py:30-40", "conduit/user/views.py", 30, 40), retrieved_docs)
        assert verify_citation(ParsedCitation("views.py:28-38", "conduit/user/views.py", 28, 38), retrieved_docs)

    def test_verify_citation_rejects_wrong_file_and_out_of_range(self, retrieved_docs):
        assert not verify_citation(ParsedCitation("fake.py:30-40", "conduit/fake.py", 30, 40), retrieved_docs)
        assert not verify_citation(ParsedCitation("views.py:80-95", "conduit/user/views.py", 80, 95), retrieved_docs)

    def test_verify_all_citations_splits_valid_and_hallucinated(self, retrieved_docs):
        answer = "Login at `conduit/user/views.py:30-40`. DB at `conduit/database.py:1-50`."
        valid, hallucinated = verify_all_citations(answer, retrieved_docs)
        assert valid == ["conduit/user/views.py:30-40"]
        assert hallucinated == ["conduit/database.py:1-50"]
