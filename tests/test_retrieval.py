"""
Tests for consolidated LangChain retrieval (src/retrieval/retrieval.py).

Covers:
  - Identifier-aware tokenization (snake_case, camelCase, PascalCase, SCREAMING)
  - BM25Retriever with custom tokenize_code preprocessing
  - Hybrid EnsembleRetriever (RRF fusion + top_k slicing)
  - CrossEncoderReranker via ContextualCompressionRetriever + graceful fallback
  - get_retriever() factory routing
"""

from unittest.mock import MagicMock, patch
import pytest

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.retrieval import retrieval
from src.retrieval.retrieval import (
    clear_retriever_cache,
    get_bm25_retriever,
    get_hybrid_retriever,
    get_rerank_retriever,
    get_retriever,
    split_identifier,
    tokenize_code,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

class FakeRetriever(BaseRetriever):
    """Minimal LangChain retriever returning a fixed list of Documents."""

    docs: list[Document]

    def _get_relevant_documents(self, query: str, **kwargs) -> list[Document]:
        return self.docs


def _make_doc(content: str, file_path: str, symbol: str, start: int = 1, end: int = 10) -> Document:
    return Document(
        page_content=content,
        metadata={
            "file_path": file_path,
            "start_line": start,
            "end_line": end,
            "symbol_name": symbol,
            "chunk_type": "function",
            "language": "python",
        },
    )


# ── 1. Tokenization Tests ────────────────────────────────────────────────────

class TestSplitIdentifier:
    def test_snake_case(self):
        assert split_identifier("get_user_by_id") == ["get", "user", "by", "id"]

    def test_camel_case(self):
        assert split_identifier("getUserById") == ["get", "user", "by", "id"]

    def test_pascal_case(self):
        assert split_identifier("UserProfileView") == ["user", "profile", "view"]

    def test_screaming_snake_case(self):
        assert split_identifier("MAX_FILE_SIZE_KB") == ["max", "file", "size", "kb"]

    def test_single_word(self):
        assert split_identifier("hello") == ["hello"]


class TestTokenizeCode:
    def test_preserves_full_identifier_and_subwords(self):
        tokens = tokenize_code("def get_user_by_id(user_id):")
        assert "get_user_by_id" in tokens
        assert "user_id" in tokens
        assert "get" in tokens
        assert "user" in tokens
        assert "id" in tokens

    def test_camel_case_tokens(self):
        tokens = tokenize_code("class UserFactory:")
        assert "userfactory" in tokens
        assert "user" in tokens
        assert "factory" in tokens

    def test_empty_and_whitespace(self):
        assert tokenize_code("") == []
        assert tokenize_code("   \n\t  ") == []


# ── 2. BM25 Retriever Tests ──────────────────────────────────────────────────

class TestBM25Retriever:
    @pytest.fixture(autouse=True)
    def reset_cache(self):
        clear_retriever_cache()
        yield
        clear_retriever_cache()

    @pytest.fixture
    def sample_docs(self) -> list[Document]:
        return [
            _make_doc(
                "def authenticate_user(username, password):\n    return check_hash(password)",
                "auth/service.py",
                "authenticate_user",
            ),
            _make_doc(
                "def create_article(title, body):\n    return db.save(Article(title, body))",
                "articles/views.py",
                "create_article",
            ),
            _make_doc(
                "JWT_SECRET_KEY = 'super-secret-key'\nTOKEN_EXPIRY = 3600",
                "config.py",
                "module",
            ),
        ]

    def test_exact_identifier_match(self, sample_docs):
        with patch.object(retrieval, "load_documents", return_value=sample_docs):
            bm25 = get_bm25_retriever("demo_repo", top_k=2)
            results = bm25.invoke("authenticate_user")

        assert len(results) == 1
        assert results[0].metadata["symbol_name"] == "authenticate_user"

    def test_subword_query_matches_compound_identifier(self, sample_docs):
        with patch.object(retrieval, "load_documents", return_value=sample_docs):
            bm25 = get_bm25_retriever("demo_repo", top_k=2)
            results = bm25.invoke("article")

        assert len(results) == 1
        assert results[0].metadata["symbol_name"] == "create_article"

    def test_zero_score_documents_are_filtered_out(self, sample_docs):
        with patch.object(retrieval, "load_documents", return_value=sample_docs):
            bm25 = get_bm25_retriever("demo_repo", top_k=5)
            results = bm25.invoke("nonexistent_symbol_xyz")

        assert results == []

    def test_missing_docs_raises_file_not_found(self):
        with patch.object(retrieval, "load_documents", return_value=[]):
            with pytest.raises(FileNotFoundError):
                get_bm25_retriever("missing_repo")


# ── 3. Hybrid Ensemble Retriever Tests ───────────────────────────────────────

class TestHybridRetriever:
    def test_rrf_boosts_shared_document_and_slices_top_k(self):
        doc_a = _make_doc("def shared(): pass", "a.py", "shared")
        doc_b = _make_doc("def vec_only(): pass", "b.py", "vec_only")
        doc_c = _make_doc("def bm25_only(): pass", "c.py", "bm25_only")

        vec_ret = FakeRetriever(docs=[doc_b, doc_a])
        bm25_ret = FakeRetriever(docs=[doc_c, doc_a])

        with (
            patch.object(retrieval, "get_vector_retriever", return_value=vec_ret) as mock_vec,
            patch.object(retrieval, "get_bm25_retriever", return_value=bm25_ret) as mock_bm25,
        ):
            hybrid = get_hybrid_retriever("demo_repo", top_k_retrieval=20, top_k_final=2)
            results = hybrid.invoke("test query")

        # Both sub-retrievers asked for 20 candidates
        mock_vec.assert_called_once_with("demo_repo", top_k=20)
        mock_bm25.assert_called_once_with("demo_repo", top_k=20)
        # Sliced to top_k_final=2 and shared doc_a wins RRF
        assert len(results) == 2
        assert results[0].metadata["symbol_name"] == "shared"


# ── 4. Cross-Encoder Rerank Retriever Tests ──────────────────────────────────

class TestRerankRetriever:
    def test_reranker_reorders_by_cross_encoder_score(self):
        doc_a = _make_doc("def low_relevance(): pass", "a.py", "low_relevance")
        doc_b = _make_doc("def high_relevance(): pass", "b.py", "high_relevance")
        doc_c = _make_doc("def mid_relevance(): pass", "c.py", "mid_relevance")

        fake_hybrid = FakeRetriever(docs=[doc_a, doc_b, doc_c])

        # Stub BaseCrossEncoder with score() returning [0.1, 0.95, 0.5]
        from langchain_classic.retrievers.document_compressors.cross_encoder import BaseCrossEncoder

        class FakeCrossEncoder(BaseCrossEncoder):
            def score(self, text_pairs):
                return [0.1, 0.95, 0.5]

        with (
            patch.object(retrieval, "get_cross_encoder", return_value=FakeCrossEncoder()),
            patch.object(retrieval, "get_hybrid_retriever", return_value=fake_hybrid),
        ):
            rerank_ret = get_rerank_retriever("demo_repo", top_k_retrieval=20, top_k_final=2)
            results = rerank_ret.invoke("find high relevance")

        assert len(results) == 2
        assert results[0].metadata["symbol_name"] == "high_relevance"
        assert results[1].metadata["symbol_name"] == "mid_relevance"

    def test_falls_back_to_hybrid_when_cross_encoder_unavailable(self):
        doc_a = _make_doc("def first(): pass", "a.py", "first")
        fake_hybrid = FakeRetriever(docs=[doc_a])

        with (
            patch.object(retrieval, "get_cross_encoder", return_value=None),
            patch.object(retrieval, "get_hybrid_retriever", return_value=fake_hybrid) as mock_hyb,
        ):
            ret = get_rerank_retriever("demo_repo", top_k_retrieval=20, top_k_final=4)
            results = ret.invoke("query")

        mock_hyb.assert_called_once_with("demo_repo", top_k_retrieval=20, top_k_final=4)
        assert results[0].metadata["symbol_name"] == "first"


# ── 5. Factory Routing Tests ─────────────────────────────────────────────────

class TestGetRetrieverFactory:
    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown retrieval mode"):
            get_retriever("demo_repo", mode="invalid_mode")

    def test_routes_all_valid_modes(self):
        sentinel = MagicMock()
        with (
            patch.object(retrieval, "get_vector_retriever", return_value=sentinel) as m_vec,
            patch.object(retrieval, "get_bm25_retriever", return_value=sentinel) as m_bm25,
            patch.object(retrieval, "get_hybrid_retriever", return_value=sentinel) as m_hyb,
            patch.object(retrieval, "get_rerank_retriever", return_value=sentinel) as m_rerank,
        ):
            assert get_retriever("r", mode="vector", top_k=4) is sentinel
            assert get_retriever("r", mode="bm25", top_k=4) is sentinel
            assert get_retriever("r", mode="hybrid", top_k=4, top_k_retrieval=20) is sentinel
            assert get_retriever("r", mode="hybrid_rerank", top_k=4, top_k_retrieval=20) is sentinel

        m_vec.assert_called_once_with("r", top_k=4)
        m_bm25.assert_called_once_with("r", top_k=4)
        m_hyb.assert_called_once_with("r", top_k_retrieval=20, top_k_final=4)
        m_rerank.assert_called_once_with("r", top_k_retrieval=20, top_k_final=4)
