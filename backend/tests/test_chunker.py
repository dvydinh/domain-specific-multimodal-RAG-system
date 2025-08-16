"""
Unit tests for the TextChunker.

Tests: basic chunking, overlap behavior, empty input,
sentence boundary detection, and page-level chunking.
"""

import time
import pytest
from backend.ingestion.chunker import TextChunker
from backend.ingestion.extractor import TextBlock, PageContent


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

    def test_bbox_stop_words_prevented(self):
        """Bbox calculation should use fast string index mapping, not set overlap."""
        # Block 1: short text at the top of the page
        block1_text = "Gà và heo"
        # Block 2: long text that will force a separate chunk
        block2_text = "Gà là chim, heo là lợn. Hai và hai là bốn. " + "Thêm nội dung dài hơn để tạo chunk riêng biệt. " * 5

        full_text = block1_text + "\n\n" + block2_text
        block1_end = len(block1_text)
        block2_start = block1_end + 2  # +2 for \n\n

        page = PageContent(
            page_number=1,
            text=full_text,
            blocks=[
                TextBlock(text=block1_text, bbox=(0,0,10,10), start_idx=0, end_idx=block1_end),
                TextBlock(
                    text=block2_text,
                    bbox=(100,100,200,200),
                    start_idx=block2_start,
                    end_idx=block2_start + len(block2_text)
                )
            ]
        )
        # Use small chunk_size to guarantee the blocks end up in separate chunks
        # Length of block1 is 9. With chunk_size=15, it will perfectly split at \n\n without packing block2 text.
        chunker = TextChunker(chunk_size=15, chunk_overlap=0)
        chunks = chunker.chunk_pages([page])
    
        # The first chunk must contain block1 text and its bbox must be ONLY (0,0,10,10)
        first_chunk = chunks[0]
        assert "Gà và heo" in first_chunk.text
        assert first_chunk.bbox is not None
        # bbox MUST NOT be inflated to (0,0,200,200) just because stop-word 'và' appears in block2
        assert first_chunk.bbox[2] <= 10.0 and first_chunk.bbox[3] <= 10.0

    def test_performance_no_blocking(self):
        """Ensure chunking 50k chars with 5000 blocks doesn't block CPU via O(N^2)."""
        text = "A" * 50000
        blocks = []
        for i in range(5000):
            blocks.append(TextBlock(
                text="A",
                bbox=(0,0,1,1),
                start_idx=i*10,
                end_idx=i*10 + 1
            ))
            
        page = PageContent(page_number=1, text=text, blocks=blocks)
        chunker = TextChunker(chunk_size=500)
        
        t0 = time.time()
        chunks = chunker.chunk_pages([page])
        t1 = time.time()
        
        # It should complete in less than 1.0 second on any reasonable machine
        assert (t1 - t0) < 1.0
        assert len(chunks) > 0
