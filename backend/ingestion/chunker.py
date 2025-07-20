"""
Text chunking with configurable size and overlap.

Implements semantic chunking using Langchain's RecursiveCharacterTextSplitter
to avoid splitting mid-sentence or mid-list when possible.
"""

import logging
from backend.models import ChunkMetadata
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class TextChunker:
    """
    Splits text into overlapping chunks for vector embedding.

    The chunker uses RecursiveCharacterTextSplitter to split at semantic 
    boundaries (paragraphs, newlines, spaces) to preserve structured data
    like ingredients lists and tables.

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
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

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
        splits = self.splitter.split_text(text)
        
        chunks: list[ChunkMetadata] = []
        for i, split in enumerate(splits):
            chunk_text = split.strip()
            if chunk_text:
                chunks.append(ChunkMetadata(
                    text=chunk_text,
                    source_pdf=source_pdf,
                    page_number=page_number,
                    chunk_index=i,
                ))

        logger.debug(
            f"Chunked {len(text)} chars into {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

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
        chunk_idx_offset = 0

        for page in pages:
            # Handle both PageContent objects and old dict format
            page_text = page.text if hasattr(page, "text") else page.get("text", "")
            page_number = page.page_number if hasattr(page, "page_number") else page.get("page_number", 0)
            blocks = page.blocks if hasattr(page, "blocks") else []

            if not page_text.strip():
                continue

            page_chunks = self.chunk_text(
                text=page_text,
                source_pdf=source_pdf,
                page_number=page_number,
            )
            
            # Map chunk bbox using blocks
            for chunk in page_chunks:
                chunk.chunk_index += chunk_idx_offset
                
                if blocks:
                    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
                    found_overlap = False
                    
                    # Split chunk text into words to find overlap
                    chunk_words = set(chunk.text.lower().split())
                    
                    for b in blocks:
                        block_words = set(b.text.lower().split())
                        
                        # If block is subset of chunk, or chunk is subset of block, or high word overlap
                        if (b.text in chunk.text or chunk.text in b.text or 
                            len(block_words & chunk_words) > 3): 
                            
                            found_overlap = True
                            bx0, by0, bx1, by1 = b.bbox
                            min_x = min(min_x, bx0)
                            min_y = min(min_y, by0)
                            max_x = max(max_x, bx1)
                            max_y = max(max_y, by1)
                            
                    if found_overlap:
                        chunk.bbox = (min_x, min_y, max_x, max_y)
            
            chunk_idx_offset += len(page_chunks)
            all_chunks.extend(page_chunks)

        logger.info(
            f"Total chunks from {len(pages)} pages: {len(all_chunks)}"
        )
        return all_chunks
