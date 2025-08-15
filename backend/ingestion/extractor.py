"""
PDF text and image extraction using PyMuPDF (fitz).

Reads each page of a PDF, extracts text blocks and embedded images,
saving images to the configured output directory with deterministic
filenames for reproducibility.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass, field

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """A block of text with its bounding box coordinates."""
    text: str
    bbox: tuple[float, float, float, float]
    start_idx: int = 0
    end_idx: int = 0


@dataclass
class PageContent:
    """Extracted content from a single PDF page."""
    page_number: int
    text: str
    blocks: list[TextBlock] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)


class PDFExtractor:
    """
    Extracts text and images from PDF files.

    Usage:
        extractor = PDFExtractor(image_output_dir="data/images")
        pages = extractor.extract("data/raw/cookbook.pdf")
    """

    def __init__(self, image_output_dir: str = "data/images"):
        self.image_output_dir = Path(image_output_dir)
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, pdf_path: str): # Generator yielding PageContent
        """
        Extract all text and images from a PDF file.

        Args:
            pdf_path: Path to the input PDF file.

        Yields:
            PageContent objects, one per page.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pdf_name = pdf_path.stem

        logger.info(f"Opening PDF: {pdf_path}")
        doc = fitz.open(str(pdf_path))

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # --- Extract text and blocks ---
            blocks = []
            full_text_parts = []
            current_idx = 0
            
            for b in page.get_text("blocks"):
                if b[6] == 0:  # Text block
                    block_text = b[4].strip()
                    if block_text:
                        start_idx = current_idx
                        end_idx = current_idx + len(block_text)
                        
                        blocks.append(TextBlock(
                            text=block_text,
                            bbox=(b[0], b[1], b[2], b[3]),
                            start_idx=start_idx,
                            end_idx=end_idx
                        ))
                        full_text_parts.append(block_text)
                        current_idx += len(block_text) + 2  # +2 for \n\n separator
            
            text = "\n\n".join(full_text_parts)

            # --- Extract images ---
            image_paths = self._extract_images(page, pdf_name, page_num)

            if text or image_paths:
                yield PageContent(
                    page_number=page_num,
                    text=text,
                    blocks=blocks,
                    image_paths=image_paths,
                )

        doc.close()
        logger.info(f"Finished extracting content from {pdf_path.name}")

    def _extract_images(
        self, page: fitz.Page, pdf_name: str, page_num: int
    ) -> list[str]:
        """Extract and save all images from a PDF page."""
        image_list = page.get_images(full=True)
        saved_paths: list[str] = []

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]

            try:
                base_image = page.parent.extract_image(xref)
                if base_image is None:
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image.get("ext", "png")

                # Deterministic filename: <pdf>_p<page>_img<index>.<ext>
                filename = f"{pdf_name}_p{page_num}_img{img_idx}.{image_ext}"
                output_path = self.image_output_dir / filename

                with open(output_path, "wb") as f:
                    f.write(image_bytes)

                saved_paths.append(str(output_path))
                logger.debug(f"Saved image: {filename}")

            except Exception as e:
                logger.warning(
                    f"Failed to extract image {img_idx} from page {page_num}: {e}"
                )

        return saved_paths

    def extract_directory(self, directory: str) -> dict[str, list[PageContent]]:
        """
        Extract from all PDFs in a directory.

        Returns:
            Dict mapping PDF filename to its extracted pages.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        results: dict[str, list[PageContent]] = {}
        pdf_files = sorted(dir_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return results

        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")

        for pdf_path in pdf_files:
            try:
                pages = self.extract(str(pdf_path))
                results[pdf_path.name] = pages
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")

        return results
