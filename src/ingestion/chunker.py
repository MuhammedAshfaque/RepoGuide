"""Code-aware chunker: Python AST chunking with 50-line overlapping fallback -> LangChain Documents."""

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
    """Create a LangChain Document with scalar Chroma-safe code metadata."""
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
            "decorators": ", ".join(decorators) if decorators else "",
            "has_docstring": has_docstring,
        },
    )


def _get_source_lines(lines: list[str], start: int, end: int) -> str:
    """Slice 1-based inclusive line range from pre-split source lines."""
    return "".join(lines[max(1, start) - 1 : min(len(lines), end)])


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Extract readable decorator names (e.g. 'app.route', 'staticmethod')."""
    names: list[str] = []
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, (ast.Name, ast.Attribute)):
            names.append(ast.unparse(target))
    return names


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """Return True if the first body statement is a string literal docstring."""
    return bool(
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    )


def _start_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """Return the node's first line including any decorators."""
    return node.decorator_list[0].lineno if node.decorator_list else node.lineno


def _chunk_class(node: ast.ClassDef, lines: list[str], file_path: str) -> list[Document]:
    """Extract method chunks plus either a class signature chunk or the whole class."""
    docs: list[Document] = []
    class_start = _start_line(node)
    class_end = node.end_lineno or node.lineno
    methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    for method in methods:
        m_start = _start_line(method)
        m_end = method.end_lineno or method.lineno
        is_async = isinstance(method, ast.AsyncFunctionDef)
        docs.append(_make_doc(
            content=_get_source_lines(lines, m_start, m_end),
            file_path=file_path,
            start_line=m_start,
            end_line=m_end,
            symbol_name=method.name,
            chunk_type="async_method" if is_async else "method",
            language="python",
            parent_class=node.name,
            decorators=_decorator_names(method),
            has_docstring=_has_docstring(method),
        ))

    # Classes with methods emit a header/attributes chunk; methodless classes stay whole
    sig_end = min(_start_line(m) for m in methods) - 1 if methods else class_end
    if sig_end >= class_start:
        docs.append(_make_doc(
            content=_get_source_lines(lines, class_start, sig_end),
            file_path=file_path,
            start_line=class_start,
            end_line=sig_end,
            symbol_name=node.name,
            chunk_type="class",
            language="python",
            decorators=_decorator_names(node),
            has_docstring=_has_docstring(node),
        ))
    return docs


def _chunk_module(lines: list[str], covered: list[tuple[int, int]], file_path: str) -> Document | None:
    """Collect top-level lines (imports, constants) outside any function or class."""
    mod_lines, mod_nums = [], []
    for i, line in enumerate(lines, 1):
        if not any(s <= i <= e for s, e in covered):
            mod_lines.append(line)
            mod_nums.append(i)

    text = "".join(mod_lines).strip()
    if not text:
        return None
    return _make_doc(
        content=text,
        file_path=file_path,
        start_line=mod_nums[0],
        end_line=mod_nums[-1],
        symbol_name="module",
        chunk_type="module",
        language="python",
    )


def chunk_python_file(source: str, file_path: str) -> list[Document]:
    """Parse Python code via AST into function, class, method, and module Documents."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_text_file(source, file_path, language="python")

    lines = source.splitlines(keepends=True)
    docs: list[Document] = []
    covered: list[tuple[int, int]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start, end = _start_line(node), node.end_lineno or node.lineno
            is_async = isinstance(node, ast.AsyncFunctionDef)
            docs.append(_make_doc(
                content=_get_source_lines(lines, start, end),
                file_path=file_path,
                start_line=start,
                end_line=end,
                symbol_name=node.name,
                chunk_type="async_function" if is_async else "function",
                language="python",
                decorators=_decorator_names(node),
                has_docstring=_has_docstring(node),
            ))
            covered.append((start, end))
        elif isinstance(node, ast.ClassDef):
            docs.extend(_chunk_class(node, lines, file_path))
            covered.append((_start_line(node), node.end_lineno or node.lineno))

    if lines:
        mod_doc = _chunk_module(lines, covered, file_path)
        if mod_doc:
            docs.append(mod_doc)

    return docs


def _guess_language(file_path: str) -> str:
    """Infer a syntax language label from the file extension."""
    return _EXTENSION_TO_LANGUAGE.get(Path(file_path).suffix.lower(), "text")


def chunk_text_file(
    source: str,
    file_path: str,
    language: str = "",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Document]:
    """Split non-Python (or unparseable) files into overlapping fixed-line Documents."""
    lines = source.splitlines(keepends=True)
    total = len(lines)
    if total == 0:
        return []

    lang = language or _guess_language(file_path)
    if total <= chunk_size:
        return [_make_doc(source.rstrip(), file_path, 1, total, "", "block", lang)]

    docs: list[Document] = []
    step = chunk_size - overlap
    for start_idx in range(0, total, step):
        end_idx = min(start_idx + chunk_size, total)
        content = "".join(lines[start_idx:end_idx]).rstrip()
        if content.strip():
            docs.append(_make_doc(content, file_path, start_idx + 1, end_idx, "", "block", lang))
        if end_idx >= total:
            break
    return docs


def chunk_file(file_info: dict) -> list[Document]:
    """Read a file dict from load_files() and route to AST or fallback chunking."""
    try:
        source = file_info["path"].read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    if not source.strip():
        return []
    if file_info["extension"] == ".py":
        return chunk_python_file(source, file_info["relative_path"])
    return chunk_text_file(source, file_info["relative_path"])
