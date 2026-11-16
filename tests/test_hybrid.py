"""
Tests for keyword search and hybrid retrieval with RRF (Phase 4).

Covers:
  - Identifier-aware tokenization (snake_case, camelCase, PascalCase, SCREAMING)
  - BM25 indexing and querying on code chunks
  - Reciprocal Rank Fusion (RRF) math and ranking logic
  - Serialization (save/load) of BM25 indexes
"""

import tempfile
from pathlib import Path
import pytest

from src.ingestion.chunker import Chunk
from src.retrieval.keyword_search import (
    tokenize_code,
    split_identifier,
    BM25Index,
    build_bm25_index,
    get_bm25_index_path,
)
from src.retrieval.hybrid import compute_rrf_fusion, _chunk_key, RRF_K_DEFAULT


# ═══════════════════════════════════════════════════════════════════════════
#  1. Tokenization Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitIdentifier:
    def test_snake_case(self):
        parts = split_identifier("get_user_by_id")
        assert parts == ["get", "user", "by", "id"]

    def test_camel_case(self):
        parts = split_identifier("getUserById")
        assert parts == ["get", "user", "by", "id"]

    def test_pascal_case(self):
        parts = split_identifier("UserProfileView")
        assert parts == ["user", "profile", "view"]

    def test_screaming_snake_case(self):
        parts = split_identifier("MAX_FILE_SIZE_KB")
        assert parts == ["max", "file", "size", "kb"]

    def test_single_word(self):
        parts = split_identifier("hello")
        assert parts == ["hello"]


class TestTokenizeCode:
    def test_includes_both_full_identifier_and_subwords(self):
        tokens = tokenize_code("def get_user_by_id(user_id):")
        # Full identifier preserved
        assert "get_user_by_id" in tokens
        assert "user_id" in tokens
        # Subwords also present
        assert "get" in tokens
        assert "user" in tokens
        assert "id" in tokens

    def test_camel_case_tokens(self):
        tokens = tokenize_code("class UserFactory:")
        assert "userfactory" in tokens
        assert "user" in tokens
        assert "factory" in tokens

    def test_preserves_special_code_terms(self):
        tokens = tokenize_code("@app.route('/api/login')")
        assert "app" in tokens
        assert "route" in tokens
        assert "api" in tokens
        assert "login" in tokens

    def test_empty_string(self):
        assert tokenize_code("") == []

    def test_whitespace_only(self):
        assert tokenize_code("   \n\t  ") == []


# ═══════════════════════════════════════════════════════════════════════════
#  2. BM25 Index & Search Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBM25Search:
    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        return [
            Chunk(
                content="def authenticate_user(username, password):\n    return check_hash(password)",
                file_path="auth/service.py",
                start_line=1,
                end_line=2,
                symbol_name="authenticate_user",
                chunk_type="function",
                language="python",
            ),
            Chunk(
                content="def create_article(title, body):\n    return db.save(Article(title, body))",
                file_path="articles/views.py",
                start_line=10,
                end_line=11,
                symbol_name="create_article",
                chunk_type="function",
                language="python",
            ),
            Chunk(
                content="JWT_SECRET_KEY = 'super-secret-key'\nTOKEN_EXPIRY = 3600",
                file_path="config.py",
                start_line=1,
                end_line=2,
                symbol_name="module",
                chunk_type="module",
                language="python",
            ),
        ]

    def test_exact_identifier_match(self, sample_chunks, tmp_path: Path):
        # Build index into a temporary directory
        import src.retrieval.keyword_search as ks
        orig_data_dir = ks.DATA_DIR
        ks.DATA_DIR = tmp_path

        try:
            build_bm25_index(sample_chunks, "test_repo")
            results = ks.bm25_search("authenticate_user", "test_repo")

            assert len(results) > 0
            # authenticate_user should be top hit
            assert results[0]["symbol_name"] == "authenticate_user"
            assert results[0]["file_path"] == "auth/service.py"
        finally:
            ks.DATA_DIR = orig_data_dir

    def test_subword_query_matches_compound_identifier(self, sample_chunks, tmp_path: Path):
        import src.retrieval.keyword_search as ks
        orig_data_dir = ks.DATA_DIR
        ks.DATA_DIR = tmp_path

        try:
            build_bm25_index(sample_chunks, "test_repo")
            # Query with subword 'article' should match create_article
            results = ks.bm25_search("article", "test_repo")

            assert len(results) > 0
            assert results[0]["symbol_name"] == "create_article"
        finally:
            ks.DATA_DIR = orig_data_dir

    def test_no_match_returns_empty(self, sample_chunks, tmp_path: Path):
        import src.retrieval.keyword_search as ks
        orig_data_dir = ks.DATA_DIR
        ks.DATA_DIR = tmp_path

        try:
            build_bm25_index(sample_chunks, "test_repo")
            results = ks.bm25_search("nonexistent_symbol_xyz", "test_repo")
            assert results == []
        finally:
            ks.DATA_DIR = orig_data_dir

    def test_index_save_and_load(self, sample_chunks, tmp_path: Path):
        import src.retrieval.keyword_search as ks
        orig_data_dir = ks.DATA_DIR
        ks.DATA_DIR = tmp_path

        try:
            path = build_bm25_index(sample_chunks, "test_repo")
            loaded = BM25Index.load(path)
            assert len(loaded.chunk_records) == 3
            assert loaded.chunk_records[0]["symbol_name"] == "authenticate_user"
        finally:
            ks.DATA_DIR = orig_data_dir


# ═══════════════════════════════════════════════════════════════════════════
#  3. Reciprocal Rank Fusion (RRF) Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRRFFusion:
    def test_item_ranked_first_in_both_lists_wins(self):
        item_a = {"file_path": "a.py", "start_line": 1, "end_line": 10, "symbol_name": "foo"}
        item_b = {"file_path": "b.py", "start_line": 1, "end_line": 10, "symbol_name": "bar"}
        item_c = {"file_path": "c.py", "start_line": 1, "end_line": 10, "symbol_name": "baz"}

        # List 1: A (#1), B (#2)
        # List 2: A (#1), C (#2)
        list_1 = [dict(item_a), dict(item_b)]
        list_2 = [dict(item_a), dict(item_c)]

        fused = compute_rrf_fusion([list_1, list_2], rrf_k=60)

        # A is #1 in both lists -> score = 1/(60+1) + 1/(60+1) = 2/61 ~= 0.032787
        assert len(fused) == 3
        assert fused[0]["symbol_name"] == "foo"
        expected_score = round(2.0 / 61.0, 6)
        assert fused[0]["rrf_score"] == expected_score
        assert fused[0]["vector_rank"] == 1
        assert fused[0]["bm25_rank"] == 1

    def test_disjoint_lists(self):
        item_a = {"file_path": "a.py", "start_line": 1, "end_line": 10, "symbol_name": "foo"}
        item_b = {"file_path": "b.py", "start_line": 1, "end_line": 10, "symbol_name": "bar"}

        list_1 = [dict(item_a)]  # Only in list 1 (#1)
        list_2 = [dict(item_b)]  # Only in list 2 (#1)

        fused = compute_rrf_fusion([list_1, list_2], rrf_k=60)

        # Both have rank 1 in their respective lists -> identical scores 1/(60+1)
        assert len(fused) == 2
        assert fused[0]["rrf_score"] == fused[1]["rrf_score"]
        # One has vector_rank=1, bm25_rank=None; the other vice-versa
        assert fused[0]["vector_rank"] == 1 or fused[0]["bm25_rank"] == 1

    def test_custom_k(self):
        item_a = {"file_path": "a.py", "start_line": 1, "end_line": 10, "symbol_name": "foo"}
        fused = compute_rrf_fusion([[dict(item_a)]], rrf_k=10)

        # 1 / (10 + 1) = 1/11 ~= 0.090909
        expected = round(1.0 / 11.0, 6)
        assert fused[0]["rrf_score"] == expected

    def test_empty_lists(self):
        fused = compute_rrf_fusion([[], []])
        assert fused == []

    def test_chunk_key_uniqueness(self):
        item_1 = {"file_path": "app.py", "start_line": 1, "end_line": 10}
        item_2 = {"file_path": "app.py", "start_line": 11, "end_line": 20}
        item_3 = {"file_path": "app.py", "start_line": "1", "end_line": "10"}

        assert _chunk_key(item_1) != _chunk_key(item_2)
        assert _chunk_key(item_1) == _chunk_key(item_3)
