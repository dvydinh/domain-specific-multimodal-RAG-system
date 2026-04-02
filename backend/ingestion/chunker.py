"""
Text chunking with configurable size and overlap.

Implements semantic chunking using Langchain's RecursiveCharacterTextSplitter
to avoid splitting mid-sentence or mid-list when possible.

Bbox mapping uses an O(n) two-pointer algorithm over pre-indexed TextBlocks
from the PDF extractor — no string search, no heuristics.
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
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True
        )

    def chunk_text(
        self,
        text: str,
        source_pdf: str = "",
        page_number: int = 0,
    ) -> list[ChunkMetadata]:
        """
        Split a text string into overlapping chunks while preserving character offsets.
        """
        if not text or not text.strip():
            return []

        # create_documents returns a list of Document objects with metadata["start_index"]
        docs = self.splitter.create_documents([text])

        chunks: list[ChunkMetadata] = []
        for i, doc in enumerate(docs):
            chunk_text = doc.page_content.strip()
            if chunk_text:
                # Store the absolute start_index from the splitter
                start_idx = doc.metadata.get("start_index", 0)
                chunks.append(ChunkMetadata(
                    text=chunk_text,
                    source_pdf=source_pdf,
                    page_number=page_number,
                    chunk_index=i,
                    start_index=start_idx  # NEW: Preserve offset for accurate Bbox lookup
                ))

        return chunks

    def chunk_pages(
        self,
        pages: list,
        source_pdf: str = "",
    ) -> list[ChunkMetadata]:
        """
        Chunk text from multiple pages with O(n) bbox mapping.

        Algorithm:
          1. Langchain splits the page text into chunks (as before).
          2. Since the extractor builds page text as '\\n\\n'.join(block_texts)
             and tracks start_idx/end_idx for each block, and chunks are
             contiguous substrings of page text, we compute each chunk's
             character interval [chunk_start, chunk_end) in the page text.
          3. A single forward pass over the sorted blocks list (two-pointer)
             collects all blocks whose [start_idx, end_idx) overlaps with
             the chunk interval, merging their bboxes.

        Complexity: O(B + C) per page where B = blocks, C = chunks.
        Total across all pages: O(n) where n = total blocks + total chunks.

        Args:
            pages: List of PageContent objects (or dicts with text/page_number/blocks).
            source_pdf: Source PDF filename.

        Returns:
            Combined list of ChunkMetadata across all pages.
        """
        all_chunks: list[ChunkMetadata] = []
        chunk_idx_offset = 0

        for page in pages:
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

            # --- O(n) Bbox mapping via two-pointer ---
            if blocks:
                # Compute each chunk's character interval in page_text.
                # Langchain splitter creates_documents preserving order, so
                # we walk page_text forward to locate each chunk's position.
                cursor = 0
                block_ptr = 0
                num_blocks = len(blocks)

                for chunk in page_chunks:
                    # SECURE FIX: Use the preserved start_index from the splitter
                    # instead of naive find() string matching.
                    chunk_start = getattr(chunk, "start_index", 0)
                    chunk_end = chunk_start + len(chunk.text)

                    # Advance block_ptr to first block that could overlap
                    # (skip blocks that end before chunk starts)
                    while block_ptr < num_blocks and blocks[block_ptr].end_idx <= chunk_start:
                        block_ptr += 1

                    # Collect bboxes from all overlapping blocks
                    min_x, min_y = float('inf'), float('inf')
                    max_x, max_y = float('-inf'), float('-inf')
                    found = False

                    # Scan from block_ptr forward; stop when block starts past chunk end
                    scan = block_ptr
                    while scan < num_blocks and blocks[scan].start_idx < chunk_end:
                        b = blocks[scan]
                        if b.end_idx > chunk_start:  # overlap confirmed
                            found = True
                            bx0, by0, bx1, by1 = b.bbox
                            min_x = min(min_x, bx0)
                            min_y = min(min_y, by0)
                            max_x = max(max_x, bx1)
                            max_y = max(max_y, by1)
                        scan += 1

                    if found:
                        chunk.bbox = (min_x, min_y, max_x, max_y)

            # Update global chunk indices
            for chunk in page_chunks:
                chunk.chunk_index += chunk_idx_offset

            chunk_idx_offset += len(page_chunks)
            all_chunks.extend(page_chunks)

        logger.info(
            f"Total chunks from {len(pages)} pages: {len(all_chunks)}"
        )
        return all_chunks
