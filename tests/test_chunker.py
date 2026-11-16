"""
Tests for the code-aware chunker (src/ingestion/chunker.py).

Covers:
  - Basic function extraction
  - Class with methods
  - Async functions and methods
  - Decorators
  - Docstrings
  - Nested functions (inner functions inside outer ones)
  - Empty files
  - Syntax errors (graceful fallback to line-based)
  - Module-level code extraction
  - Fallback line-based chunking for non-Python files
  - The chunk_file() router
"""

import textwrap
from pathlib import Path

import pytest

from src.ingestion.chunker import (
    Chunk,
    chunk_python_file,
    chunk_text_file,
    chunk_file,
    _decorator_names,
    _has_docstring,
    _guess_language,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
)


# ═══════════════════════════════════════════════════════════════════════════
#  AST Chunker Tests — Python files
# ═══════════════════════════════════════════════════════════════════════════

class TestBasicFunction:
    """Test extraction of simple top-level functions."""

    SOURCE = textwrap.dedent("""\
        def greet(name):
            return f"Hello, {name}!"
    """)

    def test_extracts_one_function(self):
        chunks = chunk_python_file(self.SOURCE, "app.py")
        func_chunks = [c for c in chunks if c.chunk_type == "function"]
        assert len(func_chunks) == 1

    def test_function_name(self):
        chunks = chunk_python_file(self.SOURCE, "app.py")
        func = [c for c in chunks if c.chunk_type == "function"][0]
        assert func.symbol_name == "greet"

    def test_line_numbers(self):
        chunks = chunk_python_file(self.SOURCE, "app.py")
        func = [c for c in chunks if c.chunk_type == "function"][0]
        assert func.start_line == 1
        assert func.end_line == 2

    def test_language_is_python(self):
        chunks = chunk_python_file(self.SOURCE, "app.py")
        func = [c for c in chunks if c.chunk_type == "function"][0]
        assert func.language == "python"

    def test_file_path_set(self):
        chunks = chunk_python_file(self.SOURCE, "src/app.py")
        func = [c for c in chunks if c.chunk_type == "function"][0]
        assert func.file_path == "src/app.py"


class TestMultipleFunctions:
    """Test a file with several functions."""

    SOURCE = textwrap.dedent("""\
        def add(a, b):
            return a + b

        def subtract(a, b):
            return a - b

        def multiply(a, b):
            return a * b
    """)

    def test_extracts_three_functions(self):
        chunks = chunk_python_file(self.SOURCE, "math_ops.py")
        funcs = [c for c in chunks if c.chunk_type == "function"]
        assert len(funcs) == 3

    def test_function_names(self):
        chunks = chunk_python_file(self.SOURCE, "math_ops.py")
        funcs = [c for c in chunks if c.chunk_type == "function"]
        names = {f.symbol_name for f in funcs}
        assert names == {"add", "subtract", "multiply"}


class TestClassWithMethods:
    """Test extraction of a class and its methods."""

    SOURCE = textwrap.dedent("""\
        class Calculator:
            \"\"\"A simple calculator.\"\"\"

            def add(self, a, b):
                return a + b

            def subtract(self, a, b):
                return a - b
    """)

    def test_extracts_class_signature(self):
        chunks = chunk_python_file(self.SOURCE, "calc.py")
        classes = [c for c in chunks if c.chunk_type == "class"]
        assert len(classes) == 1
        assert classes[0].symbol_name == "Calculator"

    def test_extracts_methods(self):
        chunks = chunk_python_file(self.SOURCE, "calc.py")
        methods = [c for c in chunks if c.chunk_type == "method"]
        assert len(methods) == 2
        names = {m.symbol_name for m in methods}
        assert names == {"add", "subtract"}

    def test_methods_have_parent_class(self):
        chunks = chunk_python_file(self.SOURCE, "calc.py")
        methods = [c for c in chunks if c.chunk_type == "method"]
        for m in methods:
            assert m.parent_class == "Calculator"

    def test_class_has_docstring(self):
        chunks = chunk_python_file(self.SOURCE, "calc.py")
        cls = [c for c in chunks if c.chunk_type == "class"][0]
        assert cls.has_docstring is True


class TestClassWithoutMethods:
    """Test a class with no methods (e.g., a config or dataclass)."""

    SOURCE = textwrap.dedent("""\
        class Config:
            DEBUG = True
            SECRET_KEY = "abc"
            DATABASE_URI = "sqlite:///db.sqlite3"
    """)

    def test_extracts_whole_class(self):
        chunks = chunk_python_file(self.SOURCE, "config.py")
        classes = [c for c in chunks if c.chunk_type == "class"]
        assert len(classes) == 1
        assert classes[0].symbol_name == "Config"
        # Should include the full class body
        assert "DEBUG" in classes[0].content
        assert "DATABASE_URI" in classes[0].content


class TestAsyncFunctions:
    """Test extraction of async functions and methods."""

    SOURCE = textwrap.dedent("""\
        import asyncio

        async def fetch_data(url):
            return await asyncio.sleep(1)

        class Client:
            async def get(self, endpoint):
                return await self._request("GET", endpoint)
    """)

    def test_async_function_type(self):
        chunks = chunk_python_file(self.SOURCE, "client.py")
        async_funcs = [c for c in chunks if c.chunk_type == "async_function"]
        assert len(async_funcs) == 1
        assert async_funcs[0].symbol_name == "fetch_data"

    def test_async_method_type(self):
        chunks = chunk_python_file(self.SOURCE, "client.py")
        async_methods = [c for c in chunks if c.chunk_type == "async_method"]
        assert len(async_methods) == 1
        assert async_methods[0].symbol_name == "get"
        assert async_methods[0].parent_class == "Client"


class TestDecorators:
    """Test that decorators are captured in metadata and included in content."""

    SOURCE = textwrap.dedent("""\
        from flask import Flask
        app = Flask(__name__)

        @app.route("/")
        def index():
            return "Hello!"

        @staticmethod
        def helper():
            pass
    """)

    def test_decorator_captured(self):
        chunks = chunk_python_file(self.SOURCE, "views.py")
        index_fn = [c for c in chunks if c.symbol_name == "index"][0]
        assert "app.route" in index_fn.decorators

    def test_staticmethod_decorator(self):
        chunks = chunk_python_file(self.SOURCE, "views.py")
        helper_fn = [c for c in chunks if c.symbol_name == "helper"][0]
        assert "staticmethod" in helper_fn.decorators

    def test_decorator_included_in_content(self):
        chunks = chunk_python_file(self.SOURCE, "views.py")
        index_fn = [c for c in chunks if c.symbol_name == "index"][0]
        assert "@app.route" in index_fn.content

    def test_decorator_start_line(self):
        chunks = chunk_python_file(self.SOURCE, "views.py")
        index_fn = [c for c in chunks if c.symbol_name == "index"][0]
        # The decorator is on line 4, the def on line 5
        assert index_fn.start_line == 4


class TestDocstrings:
    """Test that docstring presence is detected."""

    SOURCE = textwrap.dedent("""\
        def documented():
            \"\"\"This function has a docstring.\"\"\"
            pass

        def undocumented():
            pass
    """)

    def test_has_docstring_true(self):
        chunks = chunk_python_file(self.SOURCE, "funcs.py")
        doc_fn = [c for c in chunks if c.symbol_name == "documented"][0]
        assert doc_fn.has_docstring is True

    def test_has_docstring_false(self):
        chunks = chunk_python_file(self.SOURCE, "funcs.py")
        undoc_fn = [c for c in chunks if c.symbol_name == "undocumented"][0]
        assert undoc_fn.has_docstring is False


class TestNestedFunctions:
    """
    Test nested (inner) functions.

    Our chunker extracts only top-level functions and class methods.
    A nested function is part of its parent's content, not a separate chunk.
    This is intentional — the nested function is semantically part of the
    outer function's logic.
    """

    SOURCE = textwrap.dedent("""\
        def outer():
            def inner():
                return 42
            return inner()
    """)

    def test_outer_extracted(self):
        chunks = chunk_python_file(self.SOURCE, "nested.py")
        funcs = [c for c in chunks if c.chunk_type == "function"]
        assert len(funcs) == 1
        assert funcs[0].symbol_name == "outer"

    def test_inner_not_separate_chunk(self):
        chunks = chunk_python_file(self.SOURCE, "nested.py")
        names = {c.symbol_name for c in chunks if c.chunk_type == "function"}
        assert "inner" not in names

    def test_inner_included_in_outer_content(self):
        chunks = chunk_python_file(self.SOURCE, "nested.py")
        outer = [c for c in chunks if c.symbol_name == "outer"][0]
        assert "def inner" in outer.content


class TestEmptyFile:
    """Test that empty files produce no chunks."""

    def test_empty_string(self):
        chunks = chunk_python_file("", "empty.py")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = chunk_python_file("   \n\n  \n", "blank.py")
        assert chunks == []


class TestSyntaxError:
    """Test that files with syntax errors fall back to line-based chunking."""

    SOURCE = "def broken(\n    this is not valid python\n    return 42\n"

    def test_returns_chunks(self):
        chunks = chunk_python_file(self.SOURCE, "broken.py")
        assert len(chunks) > 0

    def test_chunks_are_block_type(self):
        chunks = chunk_python_file(self.SOURCE, "broken.py")
        for c in chunks:
            assert c.chunk_type == "block"

    def test_language_still_python(self):
        chunks = chunk_python_file(self.SOURCE, "broken.py")
        for c in chunks:
            assert c.language == "python"


class TestModuleLevelCode:
    """Test extraction of top-level code (imports, constants, globals)."""

    SOURCE = textwrap.dedent("""\
        import os
        from pathlib import Path

        BASE_DIR = Path(__file__).parent
        DEBUG = True

        def main():
            print("running")
    """)

    def test_module_chunk_exists(self):
        chunks = chunk_python_file(self.SOURCE, "settings.py")
        modules = [c for c in chunks if c.chunk_type == "module"]
        assert len(modules) == 1

    def test_module_contains_imports(self):
        chunks = chunk_python_file(self.SOURCE, "settings.py")
        mod = [c for c in chunks if c.chunk_type == "module"][0]
        assert "import os" in mod.content
        assert "from pathlib" in mod.content

    def test_module_contains_constants(self):
        chunks = chunk_python_file(self.SOURCE, "settings.py")
        mod = [c for c in chunks if c.chunk_type == "module"][0]
        assert "BASE_DIR" in mod.content
        assert "DEBUG" in mod.content

    def test_module_does_not_contain_function(self):
        chunks = chunk_python_file(self.SOURCE, "settings.py")
        mod = [c for c in chunks if c.chunk_type == "module"][0]
        assert "def main" not in mod.content


class TestToDict:
    """Test the Chunk.to_dict() serialization method."""

    def test_to_dict_keys(self):
        chunk = Chunk(
            content="pass",
            file_path="test.py",
            start_line=1,
            end_line=1,
            symbol_name="test",
            chunk_type="function",
            language="python",
            decorators=["staticmethod"],
        )
        d = chunk.to_dict()
        expected_keys = {
            "content", "file_path", "start_line", "end_line",
            "symbol_name", "chunk_type", "language", "parent_class",
            "decorators", "has_docstring",
        }
        assert set(d.keys()) == expected_keys

    def test_decorators_serialized_as_string(self):
        chunk = Chunk(
            content="pass",
            file_path="test.py",
            start_line=1,
            end_line=1,
            symbol_name="test",
            chunk_type="function",
            language="python",
            decorators=["app.route", "login_required"],
        )
        d = chunk.to_dict()
        assert d["decorators"] == "app.route, login_required"


# ═══════════════════════════════════════════════════════════════════════════
#  Fallback Line-Based Chunker Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackChunker:
    """Test the line-based chunker for non-Python files."""

    def test_small_file_single_chunk(self):
        source = "line1\nline2\nline3\n"
        chunks = chunk_text_file(source, "readme.md")
        assert len(chunks) == 1
        assert chunks[0].chunk_type == "block"

    def test_small_file_language_detected(self):
        chunks = chunk_text_file("hello", "index.js")
        assert chunks[0].language == "javascript"

    def test_empty_file_no_chunks(self):
        chunks = chunk_text_file("", "empty.md")
        assert len(chunks) == 0

    def test_large_file_multiple_chunks(self):
        # Create a file with 120 lines → should produce multiple chunks
        lines = [f"line {i}\n" for i in range(1, 121)]
        source = "".join(lines)
        chunks = chunk_text_file(source, "big.js", chunk_size=50, overlap=10)
        assert len(chunks) > 1

    def test_chunks_have_correct_line_numbers(self):
        lines = [f"line {i}\n" for i in range(1, 101)]
        source = "".join(lines)
        chunks = chunk_text_file(source, "data.json", chunk_size=50, overlap=10)
        # First chunk: lines 1-50
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 50
        # Second chunk: lines 41-90 (overlap of 10)
        assert chunks[1].start_line == 41
        assert chunks[1].end_line == 90

    def test_overlap_works(self):
        lines = [f"LINE_{i}\n" for i in range(1, 101)]
        source = "".join(lines)
        chunks = chunk_text_file(source, "data.txt", chunk_size=50, overlap=10)
        # The last 10 lines of chunk 0 should appear at the start of chunk 1
        c0_lines = chunks[0].content.splitlines()
        c1_lines = chunks[1].content.splitlines()
        overlap_from_c0 = c0_lines[-10:]
        overlap_from_c1 = c1_lines[:10]
        assert overlap_from_c0 == overlap_from_c1


# ═══════════════════════════════════════════════════════════════════════════
#  Language Detection Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGuessLanguage:
    def test_python(self):
        assert _guess_language("app.py") == "python"

    def test_javascript(self):
        assert _guess_language("index.js") == "javascript"

    def test_typescript(self):
        assert _guess_language("app.ts") == "typescript"

    def test_markdown(self):
        assert _guess_language("README.md") == "markdown"

    def test_unknown_extension(self):
        assert _guess_language("data.xyz") == "text"


# ═══════════════════════════════════════════════════════════════════════════
#  chunk_file() Router Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestChunkFileRouter:
    """Test that chunk_file() picks the right strategy."""

    @pytest.fixture(autouse=True)
    def setup_tmpdir(self, tmp_path: Path):
        self.tmp = tmp_path

    def _make_file(self, name: str, content: str) -> dict:
        fpath = self.tmp / name
        fpath.write_text(content, encoding="utf-8")
        return {
            "path": fpath,
            "relative_path": name,
            "extension": fpath.suffix.lower(),
        }

    def test_python_file_uses_ast(self):
        info = self._make_file("app.py", "def hello():\n    pass\n")
        chunks = chunk_file(info)
        # AST chunker produces "function" chunks, not "block"
        types = {c.chunk_type for c in chunks}
        assert "function" in types

    def test_js_file_uses_fallback(self):
        info = self._make_file("app.js", "function hello() {\n  return 1;\n}\n")
        chunks = chunk_file(info)
        assert all(c.chunk_type == "block" for c in chunks)

    def test_empty_py_file_no_chunks(self):
        info = self._make_file("empty.py", "")
        chunks = chunk_file(info)
        assert len(chunks) == 0

    def test_md_file_uses_fallback(self):
        info = self._make_file("README.md", "# Hello\n\nSome text.\n")
        chunks = chunk_file(info)
        assert all(c.chunk_type == "block" for c in chunks)
        assert chunks[0].language == "markdown"

