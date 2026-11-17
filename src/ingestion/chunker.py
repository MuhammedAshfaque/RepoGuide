"""Code-aware chunker: Python AST chunking + 50-line overlapping fallback, outputting LangChain Documents."""

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


def _get_source_lines(source: str, start: int, end: int) -> str:
    """Extract 1-based inclusive line range [start, end] from source."""
    lines = source.splitlines(keepends=True)
    return "".join(lines[max(1, start) - 1 : min(len(lines), end)])


def _decorator_names(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> list[str]:
    """Return human-readable decorator names (e.g. 'app.route', 'staticmethod')."""
    names: list[str] = []
    for dec in node.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(target, (ast.Name, ast.Attribute)):
            names.append(ast.unparse(target))
    return names


def _has_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> bool:
    """Return True if the first statement in a function or class body is a docstring."""
    return bool(
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    )


def _effective_start_line(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> int:
    """Return the first line of a node including any leading decorators."""
    return node.decorator_list[0].lineno if node.decorator_list else node.lineno


def _chunk_class(node: ast.ClassDef, source: str, file_path: str) -> list[Document]:
    """Extract each method as a Document plus a class signature (or whole class if methodless)."""
    docs: list[Document] = []
    class_start = _effective_start_line(node)
    class_end = node.end_lineno or node.lineno

    methods = [item for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for item in methods:
        m_start = _effective_start_line(item)
        m_end = item.end_lineno or item.lineno
        is_async = isinstance(item, ast.AsyncFunctionDef)
        docs.append(_make_doc(
            content=_get_source_lines(source, m_start, m_end),
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

    if methods:
        sig_end = min(_effective_start_line(m) for m in methods) - 1
        if sig_end >= class_start:
            docs.append(_make_doc(
                content=_get_source_lines(source, class_start, sig_end),
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
        docs.append(_make_doc(
            content=_get_source_lines(source, class_start, class_end),
            file_path=file_path,
            start_line=class_start,
            end_line=class_end,
            symbol_name=node.name,
            chunk_type="class",
            language="python",
            decorators=_decorator_names(node),
            has_docstring=_has_docstring(node),
        ))

    return docs


def _chunk_module(source_lines: list[str], covered: list[tuple[int, int]], file_path: str) -> Document | None:
    """Collect top-level lines outside functions/classes (imports, constants) into a module Document."""
    mod_lines: list[str] = []
    mod_nums: list[int] = []
    for i, line in enumerate(source_lines, 1):
        if not any(start <= i <= end for start, end in covered):
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
    """Parse Python source via AST into function/class/method/module Documents (fallback on SyntaxError)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return chunk_text_file(source, file_path, language="python")

    docs: list[Document] = []
    covered: list[tuple[int, int]] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = _effective_start_line(node)
            end = node.end_lineno or node.lineno
            is_async = isinstance(node, ast.AsyncFunctionDef)
            docs.append(_make_doc(
                content=_get_source_lines(source, start, end),
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
            docs.extend(_chunk_class(node, source, file_path))
            covered.append((_effective_start_line(node), node.end_lineno or node.lineno))

    source_lines = source.splitlines(keepends=True)
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
    """Split a non-Python text file into overlapping fixed-line Document blocks."""
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
    """Read a file dict from loader.load_files() and chunk it via AST (.py) or line fallback."""
    try:
        source = file_info["path"].read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    if not source.strip():
        return []
    if file_info["extension"] == ".py":
        return chunk_python_file(source, file_info["relative_path"])
    return chunk_text_file(source, file_info["relative_path"])
