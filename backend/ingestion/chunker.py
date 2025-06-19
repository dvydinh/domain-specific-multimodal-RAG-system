"""
Text chunking with configurable size and overlap.

Implements character-level chunking with sentence-boundary awareness
to avoid splitting mid-sentence when possible.
"""

import logging
import re
from backend.models import ChunkMetadata

logger = logging.getLogger(__name__)

# Regex for sentence boundaries (period, question mark, exclamation, newline)
SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?\n])\s+')


class TextChunker:
    """
    Splits text into overlapping chunks for vector embedding.

    The chunker tries to split at sentence boundaries. If a clean
    sentence break is not found within the overlap region, it falls
    back to character-level splitting.

    Args:
        chunk_size: Maximum characters per chunk (default 500).
        chunk_overlap: Characters of overlap between consecutive chunks (default 50).
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"Overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        source_pdf: str = "",
        page_number: int = 0,
    ) -> list[ChunkMetadata]:
        """
        Split a text string into overlapping chunks.

        Args:
            text: The full text to chunk.
            source_pdf: Source PDF filename for metadata.
            page_number: Page number in the source PDF.

        Returns:
            List of ChunkMetadata objects with text and positional info.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        chunks: list[ChunkMetadata] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            if end >= len(text):
                # Last chunk — take everything remaining
                chunk_text = text[start:]
            else:
                # Try to find a sentence boundary near the end
                chunk_text = text[start:end]
                boundary = self._find_sentence_boundary(chunk_text)

                if boundary is not None:
                    # Split at the sentence boundary
                    chunk_text = chunk_text[:boundary]
                    end = start + boundary

            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(ChunkMetadata(
                    text=chunk_text,
                    source_pdf=source_pdf,
                    page_number=page_number,
                    chunk_index=chunk_index,
                ))
                chunk_index += 1

            if end >= len(text):
                break

            # Move start forward, accounting for overlap
            start = end - self.chunk_overlap

        logger.debug(
            f"Chunked {len(text)} chars into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def _find_sentence_boundary(self, text: str) -> int | None:
        """
        Find the last sentence boundary in the text.
        Searches in the last `chunk_overlap` characters for a clean split point.

        Returns:
            Character position of the boundary, or None if not found.
        """
        # Search in the tail region for a sentence boundary
        search_start = max(0, len(text) - self.chunk_overlap)
        search_region = text[search_start:]

        # Find all sentence boundaries in the search region
        boundaries = list(SENTENCE_BOUNDARY.finditer(search_region))

        if boundaries:
            # Use the last boundary found
            last_boundary = boundaries[-1]
            return search_start + last_boundary.start()

        return None

    def chunk_pages(
        self,
        pages: list[dict],
        source_pdf: str = "",
    ) -> list[ChunkMetadata]:
        """
        Chunk text from multiple pages, maintaining page-level metadata.

        Args:
            pages: List of dicts with 'page_number' and 'text' keys.
            source_pdf: Source PDF filename.

        Returns:
            Combined list of ChunkMetadata across all pages.
        """
        all_chunks: list[ChunkMetadata] = []

        for page in pages:
            page_chunks = self.chunk_text(
                text=page.get("text", ""),
                source_pdf=source_pdf,
                page_number=page.get("page_number", 0),
            )
            all_chunks.extend(page_chunks)

        logger.info(
            f"Total chunks from {len(pages)} pages: {len(all_chunks)}"
        )
        return all_chunks
