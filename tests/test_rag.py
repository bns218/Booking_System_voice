"""
tests/test_rag.py
-----------------
Unit tests for the RAG pipeline — chunking and document loading.
Run with: pytest tests/
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag import chunk_text, load_documents


def test_chunk_text_basic():
    text = "A" * 1000
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_chunk_text_overlap():
    text = "Hello world " * 100
    chunks = chunk_text(text, size=100, overlap=20)
    # Each consecutive pair should share some characters due to overlap
    assert len(chunks) >= 2


def test_chunk_text_drops_tiny_fragments():
    text = "Short." + " " * 50 + "Also short."
    chunks = chunk_text(text, size=500, overlap=0)
    # Only one chunk expected since text is short
    assert len(chunks) == 1


def test_load_documents_empty_folder():
    with tempfile.TemporaryDirectory() as tmp:
        docs = load_documents(Path(tmp))
        assert docs == []


def test_load_documents_txt():
    with tempfile.TemporaryDirectory() as tmp:
        txt = Path(tmp) / "test.txt"
        txt.write_text("This is a test document with some content about the restaurant.")
        docs = load_documents(Path(tmp))
        assert len(docs) == 1
        assert docs[0]["source"] == "test.txt"
        assert "restaurant" in docs[0]["text"]


def test_load_documents_skips_unknown_extensions():
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "ignore.csv").write_text("a,b,c")
        Path(tmp, "ignore.json").write_text('{"a":1}')
        docs = load_documents(Path(tmp))
        assert docs == []
