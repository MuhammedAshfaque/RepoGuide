"""Tests for the AST and fallback chunker (src/ingestion/chunker.py)."""

import textwrap
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import (
    _guess_language,
    chunk_file,
    chunk_python_file,
    chunk_text_file,
)


class TestBasicFunction:
    SOURCE = textwrap.dedent("""\
        def greet(name):
            return f"Hello, {name}!"
    """)

    def test_extracts_one_function_as_document(self):
        docs = chunk_python_file(self.SOURCE, "src/app.py")
        funcs = [d for d in docs if d.metadata["chunk_type"] == "function"]
        assert len(funcs) == 1
        fn = funcs[0]
        assert isinstance(fn, Document)
        assert fn.page_content.startswith("def greet")
        assert fn.metadata["symbol_name"] == "greet"
        assert fn.metadata["start_line"] == 1
        assert fn.metadata["end_line"] == 2
        assert fn.metadata["language"] == "python"
        assert fn.metadata["file_path"] == "src/app.py"


class TestMultipleFunctions:
    SOURCE = textwrap.dedent("""\
        def add(a, b):
            return a + b

        def subtract(a, b):
            return a - b

        def multiply(a, b):
            return a * b
    """)

    def test_extracts_all_functions(self):
        docs = chunk_python_file(self.SOURCE, "math_ops.py")
        funcs = [d for d in docs if d.metadata["chunk_type"] == "function"]
        assert len(funcs) == 3
        assert {d.metadata["symbol_name"] for d in funcs} == {"add", "subtract", "multiply"}


class TestClassWithMethods:
    SOURCE = textwrap.dedent("""\
        class Calculator:
            \"\"\"A simple calculator.\"\"\"

            def add(self, a, b):
                return a + b

            def subtract(self, a, b):
                return a - b
    """)

    def test_extracts_class_signature_and_methods(self):
        docs = chunk_python_file(self.SOURCE, "calc.py")
        classes = [d for d in docs if d.metadata["chunk_type"] == "class"]
        methods = [d for d in docs if d.metadata["chunk_type"] == "method"]

        assert len(classes) == 1
        assert classes[0].metadata["symbol_name"] == "Calculator"
        assert classes[0].metadata["has_docstring"] is True

        assert len(methods) == 2
        assert {m.metadata["symbol_name"] for m in methods} == {"add", "subtract"}
        assert all(m.metadata["parent_class"] == "Calculator" for m in methods)


class TestClassWithoutMethods:
    SOURCE = textwrap.dedent("""\
        class Config:
            DEBUG = True
            SECRET_KEY = "abc"
            DATABASE_URI = "sqlite:///db.sqlite3"
    """)

    def test_extracts_whole_class(self):
        docs = chunk_python_file(self.SOURCE, "config.py")
        classes = [d for d in docs if d.metadata["chunk_type"] == "class"]
        assert len(classes) == 1
        assert classes[0].metadata["symbol_name"] == "Config"
        assert "DEBUG" in classes[0].page_content
        assert "DATABASE_URI" in classes[0].page_content


class TestAsyncFunctions:
    SOURCE = textwrap.dedent("""\
        import asyncio

        async def fetch_data(url):
            return await asyncio.sleep(1)

        class Client:
            async def get(self, endpoint):
                return await self._request("GET", endpoint)
    """)

    def test_async_function_and_method_types(self):
        docs = chunk_python_file(self.SOURCE, "client.py")
        afuncs = [d for d in docs if d.metadata["chunk_type"] == "async_function"]
        amethods = [d for d in docs if d.metadata["chunk_type"] == "async_method"]

        assert len(afuncs) == 1 and afuncs[0].metadata["symbol_name"] == "fetch_data"
        assert len(amethods) == 1
        assert amethods[0].metadata["symbol_name"] == "get"
        assert amethods[0].metadata["parent_class"] == "Client"


class TestDecorators:
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

    def test_decorators_captured_and_included(self):
        docs = chunk_python_file(self.SOURCE, "views.py")
        index_fn = [d for d in docs if d.metadata["symbol_name"] == "index"][0]
        helper_fn = [d for d in docs if d.metadata["symbol_name"] == "helper"][0]

        assert "app.route" in index_fn.metadata["decorators"]
        assert "@app.route" in index_fn.page_content
        assert index_fn.metadata["start_line"] == 4
        assert "staticmethod" in helper_fn.metadata["decorators"]


class TestDocstrings:
    SOURCE = textwrap.dedent("""\
        def documented():
            \"\"\"This function has a docstring.\"\"\"
            pass

        def undocumented():
            pass
    """)

    def test_docstring_detection(self):
        docs = chunk_python_file(self.SOURCE, "funcs.py")
        by_name = {d.metadata["symbol_name"]: d for d in docs}
        assert by_name["documented"].metadata["has_docstring"] is True
        assert by_name["undocumented"].metadata["has_docstring"] is False


class TestNestedFunctions:
    SOURCE = textwrap.dedent("""\
        def outer():
            def inner():
                return 42
            return inner()
    """)

    def test_inner_stays_inside_outer(self):
        docs = chunk_python_file(self.SOURCE, "nested.py")
        funcs = [d for d in docs if d.metadata["chunk_type"] == "function"]
        assert len(funcs) == 1
        assert funcs[0].metadata["symbol_name"] == "outer"
        assert "def inner" in funcs[0].page_content


class TestEdgeCases:
    def test_empty_and_whitespace(self):
        assert chunk_python_file("", "empty.py") == []
        assert chunk_python_file("   \n\n  \n", "blank.py") == []

    def test_syntax_error_falls_back_to_block(self):
        docs = chunk_python_file("def broken(\n    bad python\n", "broken.py")
        assert len(docs) > 0
        assert all(d.metadata["chunk_type"] == "block" for d in docs)
        assert all(d.metadata["language"] == "python" for d in docs)


class TestModuleLevelCode:
    SOURCE = textwrap.dedent("""\
        import os
        from pathlib import Path

        BASE_DIR = Path(__file__).parent
        DEBUG = True

        def main():
            print("running")
    """)

    def test_module_chunk_contains_imports_and_constants_only(self):
        docs = chunk_python_file(self.SOURCE, "settings.py")
        mods = [d for d in docs if d.metadata["chunk_type"] == "module"]
        assert len(mods) == 1
        assert "import os" in mods[0].page_content
        assert "BASE_DIR" in mods[0].page_content
        assert "def main" not in mods[0].page_content


class TestFallbackChunker:
    def test_small_and_empty_files(self):
        assert chunk_text_file("", "empty.md") == []
        docs = chunk_text_file("line1\nline2\n", "index.js")
        assert len(docs) == 1
        assert docs[0].metadata["chunk_type"] == "block"
        assert docs[0].metadata["language"] == "javascript"

    def test_overlap_and_line_numbers(self):
        source = "".join(f"LINE_{i}\n" for i in range(1, 101))
        docs = chunk_text_file(source, "data.txt", chunk_size=50, overlap=10)
        assert len(docs) > 1
        assert (docs[0].metadata["start_line"], docs[0].metadata["end_line"]) == (1, 50)
        assert (docs[1].metadata["start_line"], docs[1].metadata["end_line"]) == (41, 90)
        assert docs[0].page_content.splitlines()[-10:] == docs[1].page_content.splitlines()[:10]


class TestGuessLanguage:
    @pytest.mark.parametrize("fname,expected", [
        ("app.py", "python"),
        ("index.js", "javascript"),
        ("app.ts", "typescript"),
        ("README.md", "markdown"),
        ("data.xyz", "text"),
    ])
    def test_language_lookup(self, fname, expected):
        assert _guess_language(fname) == expected


class TestChunkFileRouter:
    def _make_file(self, tmp_path: Path, name: str, content: str) -> dict:
        fpath = tmp_path / name
        fpath.write_text(content, encoding="utf-8")
        return {"path": fpath, "relative_path": name, "extension": fpath.suffix.lower()}

    def test_routes_python_vs_fallback_and_empty(self, tmp_path: Path):
        py_docs = chunk_file(self._make_file(tmp_path, "app.py", "def hello():\n    pass\n"))
        js_docs = chunk_file(self._make_file(tmp_path, "app.js", "function hello() { return 1; }\n"))
        md_docs = chunk_file(self._make_file(tmp_path, "README.md", "# Hello\n"))
        empty_docs = chunk_file(self._make_file(tmp_path, "empty.py", ""))

        assert "function" in {d.metadata["chunk_type"] for d in py_docs}
        assert all(d.metadata["chunk_type"] == "block" for d in js_docs)
        assert md_docs[0].metadata["language"] == "markdown"
        assert empty_docs == []
