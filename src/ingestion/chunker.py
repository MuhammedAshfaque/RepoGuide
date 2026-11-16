"""
Code-aware chunker for CodeCompass.

Two chunking strategies:

1. **AST chunking** (Python files only):
   Uses the `ast` module to extract each function, async function, class,
   and method as a separate chunk. Each chunk carries metadata: file path,
   start/end lines, symbol name, chunk type, language, and parent class.
   A special "module" chunk captures top-level code (imports, constants,
   global statements) that lives outside any function or class.

2. **Fallback line-based chunking** (all other file types):
   Splits the file into fixed-size windows (default 50 lines) with overlap
   (default 10 lines) so context isn't lost at boundaries.

Why AST chunking?
   Fixed-size chunks split code at arbitrary line boundaries, so a search
   for "get_user_by_id" might return half the function plus unrelated code.
   AST chunking guarantees every chunk is a complete semantic unit (one
   function, one class, one method), which gives the retriever and the LLM
   clean, self-contained context.
"""

import ast
import textwrap
from dataclasses import dataclass, field
from pathlib import Path


# ── Chunk data structure ─────────────────────────────────────────────────────

@dataclass
class Chunk:
    """
    One chunk of source code with its metadata.

    Attributes:
        content:      The raw source code text of the chunk.
        file_path:    Path relative to the repo root (for citations).
        start_line:   First line number (1-based).
        end_line:     Last line number (1-based, inclusive).
        symbol_name:  Name of the function/class/method, or "module" for
                      top-level code, or "" for fallback chunks.
        chunk_type:   One of: "function", "async_function", "class", "method",
                      "async_method", "module", "block" (fallback).
        language:     Programming language (e.g. "python", "javascript").
        parent_class: For methods, the name of the enclosing class. Empty string
                      for everything else.
        decorators:   List of decorator names (e.g. ["staticmethod", "app.route"]).
        has_docstring: Whether this chunk has a docstring.
    """
    content: str
    file_path: str
    start_line: int
    end_line: int
    symbol_name: str
    chunk_type: str
    language: str
    parent_class: str = ""
    decorators: list[str] = field(default_factory=list)
    has_docstring: bool = False

    def to_dict(self) -> dict:
        """Convert to a plain dict (useful for serialization / ChromaDB metadata)."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol_name": self.symbol_name,
            "chunk_type": self.chunk_type,
            "language": self.language,
            "parent_class": self.parent_class,
            "decorators": ", ".join(self.decorators),
            "has_docstring": self.has_docstring,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_source_lines(source: str, start: int, end: int) -> str:
    """
    Extract lines `start` through `end` (1-based, inclusive) from source.
    """
    lines = source.splitlines(keepends=True)
    # Clamp to valid range
    start = max(1, start)
    end = min(len(lines), end)
    return "".join(lines[start - 1 : end])


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """
    Extract human-readable decorator names from an AST node.

    Handles:
      - @staticmethod            → "staticmethod"
      - @app.route("/")          → "app.route"
      - @login_required          → "login_required"
    """
    names: list[str] = []
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            # e.g. @app.route → we want "app.route"
            parts = []
            obj = dec
            while isinstance(obj, ast.Attribute):
                parts.append(obj.attr)
                obj = obj.value
            if isinstance(obj, ast.Name):
                parts.append(obj.id)
            names.append(".".join(reversed(parts)))
        elif isinstance(dec, ast.Call):
            # e.g. @app.route("/path") → unwrap the Call to the func
            func = dec.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                parts = []
                obj = func
                while isinstance(obj, ast.Attribute):
                    parts.append(obj.attr)
                    obj = obj.value
                if isinstance(obj, ast.Name):
                    parts.append(obj.id)
                names.append(".".join(reversed(parts)))
        # If none of the above match, skip silently
    return names


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """Check if a function or class has a docstring as its first statement."""
    if node.body and isinstance(node.body[0], ast.Expr):
        if isinstance(node.body[0].value, ast.Constant):
            return True
    return False


def _effective_start_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """
    Get the real start line of a node, including its decorators.

    If the node has decorators, the first decorator's line is earlier
    than the `def`/`class` line itself.
    """
    if node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


# ── AST Chunker (Python) ────────────────────────────────────────────────────

def chunk_python_file(source: str, file_path: str) -> list[Chunk]:
    """
    Parse a Python file with `ast` and extract one chunk per
    function/class/method, plus a "module" chunk for top-level code.

    Args:
        source:    The full source code as a string.
        file_path: Relative path to the file (used in chunk metadata).

    Returns:
        A list of Chunk objects. If the file has a syntax error,
        falls back to line-based chunking automatically.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Can't parse → fall back to line-based chunking
        return chunk_text_file(source, file_path, language="python")

    chunks: list[Chunk] = []
    source_lines = source.splitlines(keepends=True)
    total_lines = len(source_lines)

    # Track which line ranges are covered by functions/classes
    # so we can extract the remaining lines as the "module" chunk.
    covered_ranges: list[tuple[int, int]] = []  # (start, end) 1-based inclusive

    # ── Top-level functions and classes ───────────────────────────────
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = _effective_start_line(node)
            end = node.end_lineno or node.lineno
            content = _get_source_lines(source, start, end)
            is_async = isinstance(node, ast.AsyncFunctionDef)

            chunks.append(Chunk(
                content=content,
                file_path=file_path,
                start_line=start,
                end_line=end,
                symbol_name=node.name,
                chunk_type="async_function" if is_async else "function",
                language="python",
                decorators=_decorator_names(node),
                has_docstring=_has_docstring(node),
            ))
            covered_ranges.append((start, end))

        elif isinstance(node, ast.ClassDef):
            # Extract the whole class as one chunk
            class_start = _effective_start_line(node)
            class_end = node.end_lineno or node.lineno

            # Also extract each method inside the class as its own chunk
            has_methods = False
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    has_methods = True
                    m_start = _effective_start_line(item)
                    m_end = item.end_lineno or item.lineno
                    m_content = _get_source_lines(source, m_start, m_end)
                    is_async = isinstance(item, ast.AsyncFunctionDef)

                    chunks.append(Chunk(
                        content=m_content,
                        file_path=file_path,
                        start_line=m_start,
                        end_line=m_end,
                        symbol_name=item.name,
                        chunk_type="async_method" if is_async else "method",
                        language="python",
                        parent_class=node.name,
                        decorators=_decorator_names(item),
                        has_docstring=_has_docstring(item),
                    ))

            # If the class has methods, also add the class "signature"
            # (the class def line + docstring + class-level attributes)
            # as a separate chunk so the retriever can find the class itself.
            # If the class has NO methods (e.g., a dataclass or config class),
            # add the entire class as one chunk.
            if has_methods:
                # Class signature: from class_start to the line before
                # the first method
                first_method_start = min(
                    _effective_start_line(item)
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                )
                sig_end = first_method_start - 1
                if sig_end >= class_start:
                    sig_content = _get_source_lines(source, class_start, sig_end)
                    chunks.append(Chunk(
                        content=sig_content,
                        file_path=file_path,
                        start_line=class_start,
                        end_line=sig_end,
                        symbol_name=node.name,
                        chunk_type="class",
                        language="python",
                        decorators=_decorator_names(node),
                        has_docstring=_has_docstring(node),
                    ))
            else:
                # No methods → add the whole class as one chunk
                class_content = _get_source_lines(source, class_start, class_end)
                chunks.append(Chunk(
                    content=class_content,
                    file_path=file_path,
                    start_line=class_start,
                    end_line=class_end,
                    symbol_name=node.name,
                    chunk_type="class",
                    language="python",
                    decorators=_decorator_names(node),
                    has_docstring=_has_docstring(node),
                ))

            covered_ranges.append((class_start, class_end))

    # ── Module-level chunk ────────────────────────────────────────────
    # Collect lines that are NOT inside any function/class.
    # These are imports, global variables, constants, etc.
    if total_lines > 0:
        module_lines: list[str] = []
        module_line_numbers: list[int] = []

        for i in range(1, total_lines + 1):
            in_covered = any(start <= i <= end for start, end in covered_ranges)
            if not in_covered:
                line = source_lines[i - 1]
                # Skip completely blank lines at the very start/end
                module_lines.append(line)
                module_line_numbers.append(i)

        # Build the module chunk from non-empty collected lines
        module_text = "".join(module_lines).strip()
        if module_text:
            chunks.append(Chunk(
                content=module_text,
                file_path=file_path,
                start_line=module_line_numbers[0] if module_line_numbers else 1,
                end_line=module_line_numbers[-1] if module_line_numbers else total_lines,
                symbol_name="module",
                chunk_type="module",
                language="python",
            ))

    return chunks


# ── Fallback Line-Based Chunker ──────────────────────────────────────────────

# Default chunk size and overlap for non-Python files.
# 50 lines is small enough to fit in context but large enough to contain
# a meaningful block of code. 10-line overlap prevents losing context
# at chunk boundaries (e.g., a function header on line 49, body on line 51).
DEFAULT_CHUNK_SIZE = 50
DEFAULT_OVERLAP = 10


def chunk_text_file(
    source: str,
    file_path: str,
    language: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """
    Split a text file into overlapping fixed-size chunks.

    Used for non-Python files (.js, .ts, .md, .json, etc.) where
    we can't do AST-based extraction.

    Args:
        source:     The full file content as a string.
        file_path:  Relative path (for metadata).
        language:   Language label (e.g. "javascript", "markdown").
        chunk_size: Number of lines per chunk.
        overlap:    Number of lines to overlap between consecutive chunks.

    Returns:
        A list of Chunk objects of type "block".
    """
    lines = source.splitlines(keepends=True)
    total = len(lines)

    if total == 0:
        return []

    # Auto-detect language from extension if not provided
    if not language:
        language = _guess_language(file_path)

    # If the file is small enough, return it as a single chunk
    if total <= chunk_size:
        return [Chunk(
            content=source.rstrip(),
            file_path=file_path,
            start_line=1,
            end_line=total,
            symbol_name="",
            chunk_type="block",
            language=language,
        )]

    chunks: list[Chunk] = []
    step = chunk_size - overlap  # how far we advance each window

    for start_idx in range(0, total, step):
        end_idx = min(start_idx + chunk_size, total)
        chunk_lines = lines[start_idx:end_idx]
        content = "".join(chunk_lines).rstrip()

        if content.strip():  # skip empty chunks
            chunks.append(Chunk(
                content=content,
                file_path=file_path,
                start_line=start_idx + 1,       # 1-based
                end_line=end_idx,                # 1-based inclusive
                symbol_name="",
                chunk_type="block",
                language=language,
            ))

        # Stop if we've reached the end
        if end_idx >= total:
            break

    return chunks


# ── Language Detection ───────────────────────────────────────────────────────

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".sh": "shell",
    ".bat": "batch",
    ".toml": "toml",
    ".txt": "text",
    ".rst": "restructuredtext",
}


def _guess_language(file_path: str) -> str:
    """Guess the language from the file extension."""
    ext = Path(file_path).suffix.lower()
    return _EXTENSION_TO_LANGUAGE.get(ext, "text")


# ── Main Entry Point ────────────────────────────────────────────────────────

def chunk_file(file_info: dict) -> list[Chunk]:
    """
    Chunk a single file, choosing the right strategy based on extension.

    Args:
        file_info: A dict from loader.load_files() with keys:
                   'path' (Path object), 'relative_path' (str), 'extension' (str)

    Returns:
        A list of Chunk objects for this file.
    """
    file_path: Path = file_info["path"]
    relative_path: str = file_info["relative_path"]
    extension: str = file_info["extension"]

    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    if not source.strip():
        return []  # Empty file → no chunks

    if extension == ".py":
        return chunk_python_file(source, relative_path)
    else:
        return chunk_text_file(source, relative_path)


# ── CLI for quick testing ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.chunker <file_path>")
        print("       python -m src.ingestion.chunker <repo_source>  (all files)")
        sys.exit(1)

    target = sys.argv[1]
    target_path = Path(target)

    if target_path.is_file():
        # Chunk a single file
        info = {
            "path": target_path,
            "relative_path": target_path.name,
            "extension": target_path.suffix.lower(),
        }
        chunks = chunk_file(info)
        print(f"\n{'='*60}")
        print(f" File: {target_path.name}  |  Chunks: {len(chunks)}")
        print(f"{'='*60}")
        for i, c in enumerate(chunks, 1):
            label = c.symbol_name or "(block)"
            parent = f" in {c.parent_class}" if c.parent_class else ""
            dec = f"  decorators: {c.decorators}" if c.decorators else ""
            print(f"\n--- Chunk {i}: {c.chunk_type} '{label}'{parent} "
                  f"[L{c.start_line}-{c.end_line}]{dec} ---")
            # Show first 5 lines of content
            preview = c.content.splitlines()[:5]
            for line in preview:
                print(f"  {line.rstrip()}")
            if len(c.content.splitlines()) > 5:
                print(f"  ... ({len(c.content.splitlines()) - 5} more lines)")
    else:
        # Assume it's a repo source → load all files, chunk them all
        from src.ingestion.loader import load_files

        files = load_files(target)
        total_chunks = 0
        for finfo in files:
            chunks = chunk_file(finfo)
            total_chunks += len(chunks)
            if chunks:
                print(f"  {finfo['relative_path']:50s} → {len(chunks):3d} chunks")

        print(f"\n{'='*60}")
        print(f" Total files: {len(files)}  |  Total chunks: {total_chunks}")
        print(f"{'='*60}")
