"""
Tests for the hybrid search pipeline (src/retrieval/hybrid.py).

This file is a placeholder created in Phase 3. The actual hybrid search
tests will be added in Phase 4 when BM25 and RRF fusion are implemented.

For Phase 3, we focus on integration testing of the indexer and vector
search via the CLI smoke tests rather than unit tests, because:
  - The embedding model and ChromaDB are external dependencies
  - Unit-testing them would require mocking, which adds complexity
    without much value at this stage
  - The CLI smoke test (index + search) covers the full pipeline
"""

# Phase 4 will add:
# - test_bm25_search
# - test_rrf_fusion
# - test_hybrid_vs_vector_comparison

