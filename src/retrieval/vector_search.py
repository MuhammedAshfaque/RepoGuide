"""
Vector search for CodeCompass.

Queries ChromaDB for chunks that are semantically similar to a
user's question. Uses the same embedding model as the indexer to
embed the query, then finds the nearest neighbors in the vector space.

How it works:
  1. The user asks: "Where is authentication handled?"
  2. We embed that question into a vector (384 dimensions for MiniLM).
  3. ChromaDB finds the stored code chunks whose embeddings are closest
     to the question embedding (using cosine similarity).
  4. We return the top-k results with their similarity scores and metadata.

Why cosine similarity?
  Cosine similarity measures the angle between two vectors, ignoring their
  magnitude. This works well for text embeddings because two texts about
  the same topic will point in similar directions in the embedding space,
  even if one is much longer than the other.
"""

from src.ingestion.indexer import get_embedding_model, get_chroma_client, _collection_name
from src.config import TOP_K_RETRIEVAL


def vector_search(
    query: str,
    repo_name: str,
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Search for code chunks that are semantically similar to the query.

    Args:
        query:     The user's natural-language question.
        repo_name: Name of the indexed repo (used to find the ChromaDB collection).
        top_k:     Number of results to return (default from config: 20).

    Returns:
        A list of dicts, each with:
          - 'content':     The chunk's source code.
          - 'file_path':   Relative path in the repo.
          - 'start_line':  First line number.
          - 'end_line':    Last line number.
          - 'symbol_name': Function/class/method name.
          - 'chunk_type':  "function", "class", "method", etc.
          - 'language':    Programming language.
          - 'parent_class': Enclosing class name (for methods).
          - 'score':       Similarity score (higher = more similar).
          - 'rank':        1-based rank position.

    Raises:
        ValueError: If the collection doesn't exist (repo not indexed).
    """
    # Get the collection
    col_name = _collection_name(repo_name)
    client = get_chroma_client()

    try:
        collection = client.get_collection(col_name)
    except Exception:
        raise ValueError(
            f"No index found for repo '{repo_name}'. "
            f"Run indexing first with: python -m src.ingestion.indexer <repo>"
        )

    # Embed the query
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    # Unpack results — ChromaDB returns nested lists (one per query)
    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    # Convert to a clean list of dicts
    # ChromaDB with cosine space returns distances (0 = identical, 2 = opposite).
    # Convert to a similarity score: score = 1 - (distance / 2), range [0, 1].
    output: list[dict] = []
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        score = 1.0 - (dist / 2.0)  # cosine distance → similarity
        output.append({
            "content": doc,
            "file_path": meta.get("file_path", ""),
            "start_line": meta.get("start_line", 0),
            "end_line": meta.get("end_line", 0),
            "symbol_name": meta.get("symbol_name", ""),
            "chunk_type": meta.get("chunk_type", ""),
            "language": meta.get("language", ""),
            "parent_class": meta.get("parent_class", ""),
            "decorators": meta.get("decorators", ""),
            "has_docstring": meta.get("has_docstring", False),
            "score": round(score, 4),
            "rank": i + 1,
        })

    return output


# ── CLI for testing ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m src.retrieval.vector_search <repo_name> <query>")
        print('Example: python -m src.retrieval.vector_search flask-realworld-example-app "Where is authentication handled?"')
        sys.exit(1)

    repo_name = sys.argv[1]
    query = " ".join(sys.argv[2:])

    print(f"\n🔍 Query: \"{query}\"")
    print(f"📦 Repo:  {repo_name}")
    print(f"{'='*70}")

    results = vector_search(query, repo_name, top_k=10)

    for r in results:
        symbol = r["symbol_name"] or "(block)"
        parent = f" in {r['parent_class']}" if r["parent_class"] else ""
        citation = f"{r['file_path']}:{r['start_line']}-{r['end_line']}"
        print(f"\n  #{r['rank']}  score={r['score']:.4f}  {r['chunk_type']} "
              f"'{symbol}'{parent}")
        print(f"       📄 {citation}")
        # Show first 3 lines of content
        preview = r["content"].splitlines()[:3]
        for line in preview:
            print(f"       {line.rstrip()}")
        if len(r["content"].splitlines()) > 3:
            print(f"       ... ({len(r['content'].splitlines()) - 3} more lines)")

