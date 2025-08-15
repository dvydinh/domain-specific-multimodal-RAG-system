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
            separators=["\\n\\n", "\\n", " ", ""],
            add_start_index=True  # Ensure LangChain metadata calculates exact offsets
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
        pages: list,  # can be list of dicts or PageContent objects
        source_pdf: str = "",
    ) -> list[ChunkMetadata]:
        """
        Chunk text from multiple pages, maintaining page-level metadata.
        Uses O(n) sweep pointer layout to natively map BBox from LangChain origins.

        Args:
            pages: List of dicts with 'page_number' and 'text' keys.
            source_pdf: Source PDF filename.

        Returns:
            Combined list of ChunkMetadata across all pages.
        """
        from langchain_core.documents import Document
        
        all_chunks: list[ChunkMetadata] = []
        chunk_idx_offset = 0

        for page in pages:
            # Handle both PageContent objects and old dict format
            page_text = page.text if hasattr(page, "text") else page.get("text", "")
            page_number = page.page_number if hasattr(page, "page_number") else page.get("page_number", 0)
            blocks = page.blocks if hasattr(page, "blocks") else []

            if not page_text.strip():
                continue

            # Load into Langchain Document to generate exact algorithmic split intervals
            doc = Document(page_content=page_text)
            splits = self.splitter.split_documents([doc])
            
            # Map chunk bbox natively using string index intervals with O(n) sweep
            current_block_idx = 0
            
            for split in splits:
                chunk_text = split.page_content.strip()
                if not chunk_text:
                    continue
                    
                chunk = ChunkMetadata(
                    text=chunk_text,
                    source_pdf=source_pdf,
                    page_number=page_number,
                    chunk_index=chunk_idx_offset,
                )
                chunk_idx_offset += 1
                
                start_idx = split.metadata.get("start_index", -1)
                
                if blocks and start_idx != -1:
                    # Extract the precise location matching the stripped text
                    # because split.page_content might contain trailing/leading separators
                    chunk_start = start_idx + split.page_content.find(chunk_text)
                    chunk_end = chunk_start + len(chunk_text)
                    
                    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
                    found_overlap = False
                    
                    # Advance current_block_idx to the first block that could intersect this chunk
                    while current_block_idx < len(blocks) and blocks[current_block_idx].end_idx <= chunk_start:
                        current_block_idx += 1
                        
                    # Now sequentially overlay until boundaries exceed the chunk
                    temp_idx = current_block_idx
                    while temp_idx < len(blocks) and blocks[temp_idx].start_idx < chunk_end:
                        b = blocks[temp_idx]
                        overlap_start = max(b.start_idx, chunk_start)
                        overlap_end = min(b.end_idx, chunk_end)
                        
                        if overlap_start < overlap_end:
                            found_overlap = True
                            bx0, by0, bx1, by1 = b.bbox
                            min_x = min(min_x, bx0)
                            min_y = min(min_y, by0)
                            max_x = max(max_x, bx1)
                            max_y = max(max_y, by1)
                            
                        temp_idx += 1
                        
                    if found_overlap:
                        chunk.bbox = (min_x, min_y, max_x, max_y)
                        
                all_chunks.append(chunk)

        logger.info(
            f"Total chunks from {len(pages)} pages: {len(all_chunks)}"
        )
        return all_chunks
