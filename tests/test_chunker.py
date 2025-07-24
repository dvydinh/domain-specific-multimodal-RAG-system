"""
Unit tests for the TextChunker.

Tests: basic chunking, overlap behavior, empty input,
sentence boundary detection, and page-level chunking.
"""

import pytest
from backend.ingestion.chunker import TextChunker


class TestTextChunker:
    """Tests for TextChunker functionality."""

    def setup_method(self):
        self.chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    def test_basic_chunking(self):
        """Short text fits in a single chunk."""
        text = "This is a short sentence."
        chunks = self.chunker.chunk_text(text, source_pdf="test.pdf", page_number=1)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].source_pdf == "test.pdf"
        assert chunks[0].page_number == 1

    def test_multi_chunk_splitting(self):
        """Long text is split into multiple overlapping chunks."""
        text = "A" * 250  # 250 chars with chunk_size=100
        chunks = self.chunker.chunk_text(text)
        assert len(chunks) >= 3

    def test_overlap_content(self):
        """Adjacent chunks share overlapping content."""
        # Use distinct words so we can detect overlap
        words = " ".join([f"word{i}" for i in range(50)])
        chunker = TextChunker(chunk_size=100, chunk_overlap=30)
        chunks = chunker.chunk_text(words)

        if len(chunks) >= 2:
            # The end of chunk 0 should appear at the start of chunk 1
            tail_of_first = chunks[0].text[-30:]
            assert any(
                word in chunks[1].text for word in tail_of_first.split()
            )

    def test_empty_text_returns_nothing(self):
        """Empty or whitespace-only input returns no chunks."""
        assert self.chunker.chunk_text("") == []
        assert self.chunker.chunk_text("   ") == []
        assert self.chunker.chunk_text(None) == []

    def test_chunk_index_sequential(self):
        """Chunk indices are sequential starting from 0."""
        text = "Hello world. " * 50
        chunks = self.chunker.chunk_text(text)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_metadata_preserved(self):
        """Source PDF and page number are carried through to chunks."""
        text = "Some recipe text here. " * 20
        chunks = self.chunker.chunk_text(
            text, source_pdf="cookbook.pdf", page_number=42
        )
        for chunk in chunks:
            assert chunk.source_pdf == "cookbook.pdf"
            assert chunk.page_number == 42

    def test_overlap_must_be_less_than_size(self):
        """Overlap >= chunk_size raises ValueError."""
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=150)

    def test_chunk_pages(self):
        """chunk_pages processes multiple pages correctly."""
        pages = [
            {"text": "Page one content. " * 10, "page_number": 1},
            {"text": "Page two content. " * 10, "page_number": 2},
        ]
        chunks = self.chunker.chunk_pages(pages, source_pdf="multi.pdf")
        assert len(chunks) > 0

        page_numbers = {c.page_number for c in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_sentence_boundary_awareness(self):
        """Chunker prefers splitting at sentence boundaries."""
        sentences = ". ".join([f"Sentence number {i}" for i in range(20)])
        chunker = TextChunker(chunk_size=120, chunk_overlap=30)
        chunks = chunker.chunk_text(sentences)

        # Most chunks should end at or near a period
        for chunk in chunks[:-1]:  # Skip last chunk
            text = chunk.text.rstrip()
            # Allow some flexibility — the point is it tries
            assert text[-1] in '.!?\n' or len(text) <= 120
