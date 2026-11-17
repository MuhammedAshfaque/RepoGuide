"""Code-aware chunker: Python AST units + 50-line overlapping fallback -> LangChain Documents."""

import ast
from pathlib import Path
from langchain_core.documents import Document

DEFAULT_CHUNK_SIZE = 50
DEFAULT_OVERLAP = 10

_EXTENSION_TO_LANGUAGE: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript", ".java": "java",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".cs": "csharp",
    ".md": "markdown", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".html": "html", ".css": "css", ".sql": "sql", ".sh": "shell",
    ".bat": "batch", ".toml": "toml", ".txt": "text", ".rst": "restructuredtext",
}


def _make_doc(
    content: str,
    file_path: str,
    start_line: int,
    end_line: int,
    symbol_name: str,
    chunk_type: str,
    language: str,
    parent_class: str = "",
    decorators: list[str] | None = None,
    has_docstring: bool = False,
) -> Document:
    """Create a LangChain Document with Chroma-safe scalar metadata."""
    return Document(
        page_content=content,
        metadata={
            "file_path": file_path,
            "start_line": start_line,
            "end_line": end_line,
            "symbol_name": symbol_name,
            "chunk_type": chunk_type,
            "language": language,
            "parent_class": parent_class,
            "decorators": ", ".join(decorators or []),
            "has_docstring": has_docstring,
        },
    )


def _slice_lines(source_lines: list[str], start: int, end: int) -> str:
    """Extract 1-based inclusive line range [start, end] from split lines."""
    return "".join(source_lines[max(1, start) - 1 : min(len(source_lines), end)])


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Return readable decorator names like 'app.route' or 'staticmethod'."""
    names: list[str] = []
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, (ast.Name, ast.Attribute)):
            names.append(ast.unparse(target))
    return names


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """Return True if the node starts with a docstring literal."""
    return bool(node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant))


def _start_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """Return the first line of a node including any leading decorators."""
    return node.decorator_list[0].lineno if node.decorator_list else node.lineno


def _chunk_class(node: ast.ClassDef, source_lines: list[str], file_path: str) -> list[Document]:
    """Extract each method as a Document plus the class header (or whole class if methodless)."""
    docs: list[Document] = []
    class_start, class_end = _start_line(node), node.end_lineno or node.lineno
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    for item in methods:
        m_start, m_end = _start_line(item), item.end_lineno or item.lineno
        docs.append(_make_doc(
            content=_slice_lines(source_lines, m_start, m_end),
            file_path=file_path, start_line=m_start, end_line=m_end,
            symbol_name=item.name,
            chunk_type="async_method" if isinstance(item, ast.AsyncFunctionDef) else "method",
            language="python", parent_class=node.name,
            decorators=_decorator_names(item), has_docstring=_has_docstring(item),
        ))

    sig_end = min(_start_line(m) for m in methods) - 1 if methods else class_end
    if sig_end >= class_start:
        docs.append(_make_doc(
            content=_slice_lines(source_lines, class_start, sig_end),
            file_path=file_path, start_line=class_start, end_line=sig_end,
            symbol_name=node.name, chunk_type="class", language="python",
            decorators=_decorator_names(node), has_docstring=_has_docstring(node),
        ))
    return docs


def _chunk_module(source_lines: list[str], covered: list[tuple[int, int]], file_path: str) -> Document | None:
    """Collect top-level lines outside functions/classes into one module Document."""
    remaining = [(i, line) for i, line in enumerate(source_lines, 1) if not any(s <= i <= e for s, e in covered)]
    text = "".join(line for _, line in remaining).strip()
    if not text:
        return None
    return _make_doc(text, file_path, remaining[0][0], remaining[-1][0], "module", "module", "python")


def chunk_python_file(source: str, file_path: str) -> list[Document]:
    """Parse Python source via AST into function, class, method, and module Documents."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_text_file(source, file_path, language="python")

    source_lines = source.splitlines(keepends=True)
    docs: list[Document] = []
    covered: list[tuple[int, int]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _start_line(node), node.end_lineno or node.lineno
            docs.append(_make_doc(
                content=_slice_lines(source_lines, start, end),
                file_path=file_path, start_line=start, end_line=end,
                symbol_name=node.name,
                chunk_type="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                language="python", decorators=_decorator_names(node), has_docstring=_has_docstring(node),
            ))
            covered.append((start, end))
        elif isinstance(node, ast.ClassDef):
            docs.extend(_chunk_class(node, source_lines, file_path))
            covered.append((_start_line(node), node.end_lineno or node.lineno))

    if source_lines:
        mod_doc = _chunk_module(source_lines, covered, file_path)
        if mod_doc:
            docs.append(mod_doc)
    return docs


def _guess_language(file_path: str) -> str:
    """Infer a language label from a file extension."""
    return _EXTENSION_TO_LANGUAGE.get(Path(file_path).suffix.lower(), "text")


def chunk_text_file(
    source: str,
    file_path: str,
    language: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Document]:
    """Split a non-Python file into overlapping fixed-line Document blocks."""
    lines = source.splitlines(keepends=True)
    total = len(lines)
    if total == 0:
        return []

    lang = language or _guess_language(file_path)
    step = max(1, chunk_size - overlap)
    docs: list[Document] = []

    for start_idx in range(0, total, step):
        end_idx = min(start_idx + chunk_size, total)
        content = "".join(lines[start_idx:end_idx]).rstrip()
        if content.strip():
            docs.append(_make_doc(content, file_path, start_idx + 1, end_idx, "", "block", lang))
        if end_idx >= total:
            break
    return docs


def chunk_file(file_info: dict) -> list[Document]:
    """Read a file dict from load_files() and chunk via AST (.py) or line fallback."""
    try:
        source = file_info["path"].read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return []
    if not source.strip():
        return []
    if file_info["extension"] == ".py":
        return chunk_python_file(source, file_info["relative_path"])
    return chunk_text_file(source, file_info["relative_path"])
