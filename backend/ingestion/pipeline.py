"""
Ingestion pipeline orchestrator.

Coordinates the full ETL flow:
  PDF → Extract → Chunk → Entity Extract → Graph Build → Vector Embed

This is the main entry point for processing new recipe PDFs.
"""

import logging
from pathlib import Path
from typing import Optional

from backend.config import get_settings
from backend.models import ChunkMetadata, ImageMetadata
from backend.ingestion.extractor import PDFExtractor, PageContent
from backend.ingestion.chunker import TextChunker
from backend.ingestion.entity_extractor import EntityExtractor
from backend.ingestion.graph_builder import GraphBuilder
from backend.ingestion.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline for recipe PDFs.

    Orchestrates:
      1. PDF extraction (text + images)
      2. Text chunking with overlap
      3. LLM entity extraction (recipes, ingredients, tags)
      4. Neo4j graph construction
      5. Vector embedding and Qdrant storage

    Usage:
        pipeline = IngestionPipeline()
        stats = pipeline.ingest("data/raw/cookbook.pdf")
        print(stats)
    """

    def __init__(
        self,
        extractor: Optional[PDFExtractor] = None,
        chunker: Optional[TextChunker] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        graph_builder: Optional[GraphBuilder] = None,
        vector_store: Optional[VectorStoreManager] = None,
    ):
        settings = get_settings()

        self.extractor = extractor or PDFExtractor(
            image_output_dir=settings.image_output_dir
        )
        self.chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.graph_builder = graph_builder or GraphBuilder()
        self.vector_store = vector_store or VectorStoreManager()

    def setup(self):
        """Initialize database schemas and collections."""
        logger.info("Setting up database schemas...")
        self.graph_builder.create_constraints()
        self.vector_store.create_collections()
        logger.info("Database schemas ready")

    def ingest(self, pdf_path: str) -> dict:
        """
        Process a single PDF file through the full pipeline.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict with ingestion statistics.
        """
        pdf_name = Path(pdf_path).name
        logger.info(f"{'=' * 60}")
        logger.info(f"Starting ingestion: {pdf_name}")
        logger.info(f"{'=' * 60}")

        stats = {
            "pdf": pdf_name,
            "pages_extracted": 0,
            "chunks_created": 0,
            "entities_extracted": 0,
            "recipes_added": 0,
            "text_vectors": 0,
            "image_vectors": 0,
        }

        # === Step 1: Extract text and images ===
        logger.info("[Step 1/5] Extracting text and images from PDF...")
        pages = self.extractor.extract(pdf_path)
        stats["pages_extracted"] = len(pages)
        logger.info(f"  → Extracted {len(pages)} pages")

        if not pages:
            logger.warning(f"No content extracted from {pdf_name}")
            return stats

        # === Step 2: Chunk text ===
        logger.info("[Step 2/5] Chunking text...")
        all_chunks = self._chunk_pages(pages, pdf_name)
        stats["chunks_created"] = len(all_chunks)
        logger.info(f"  → Created {len(all_chunks)} text chunks")

        # === Step 3: Extract entities via LLM ===
        logger.info("[Step 3/5] Extracting entities with LLM...")
        chunk_texts = [c.text for c in all_chunks]
        entities = self.entity_extractor.extract_batch(chunk_texts)
        stats["entities_extracted"] = len(entities)
        logger.info(f"  → Found {len(entities)} recipe entities")

        # === Step 4: Build knowledge graph ===
        logger.info("[Step 4/5] Building knowledge graph in Neo4j...")
        recipe_id_map = self.graph_builder.add_recipes(
            entities, source_pdf=pdf_name
        )
        stats["recipes_added"] = len(recipe_id_map)
        logger.info(f"  → Added {len(recipe_id_map)} recipes to graph")

        # Assign recipe names to chunks for cross-referencing
        self._assign_recipe_names(all_chunks, entities)

        # === Step 5: Embed and store vectors ===
        logger.info("[Step 5/5] Embedding and storing vectors in Qdrant...")

        # Text vectors
        text_count = self.vector_store.embed_and_store_chunks(
            all_chunks, recipe_id_map
        )
        stats["text_vectors"] = text_count

        # Image vectors
        image_metadata = self._collect_image_metadata(pages, pdf_name, recipe_id_map)
        image_count = self.vector_store.embed_and_store_images(
            image_metadata, recipe_id_map
        )
        stats["image_vectors"] = image_count

        logger.info(f"  → Stored {text_count} text + {image_count} image vectors")

        logger.info(f"{'=' * 60}")
        logger.info(f"Ingestion complete: {pdf_name}")
        logger.info(f"Stats: {stats}")
        logger.info(f"{'=' * 60}")

        return stats

    def ingest_directory(self, directory: str) -> list[dict]:
        """
        Process all PDFs in a directory.

        Returns:
            List of stats dicts, one per PDF.
        """
        dir_path = Path(directory)
        pdf_files = sorted(dir_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []

        logger.info(f"Found {len(pdf_files)} PDFs to process")
        all_stats = []

        for pdf_path in pdf_files:
            try:
                stats = self.ingest(str(pdf_path))
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                all_stats.append({"pdf": pdf_path.name, "error": str(e)})

        return all_stats

    # ================================================================
    # Internal Helpers
    # ================================================================

    def _chunk_pages(
        self, pages: list[PageContent], pdf_name: str
    ) -> list[ChunkMetadata]:
        """Convert extracted pages into text chunks."""
        page_dicts = [
            {"text": p.text, "page_number": p.page_number}
            for p in pages
            if p.text.strip()
        ]
        return self.chunker.chunk_pages(page_dicts, source_pdf=pdf_name)

    def _assign_recipe_names(
        self,
        chunks: list[ChunkMetadata],
        entities: list,
    ):
        """
        Best-effort assignment of recipe names to text chunks.
        Matches chunks to entities based on text content overlap.
        """
        for chunk in chunks:
            for entity in entities:
                # If the recipe name appears in the chunk text, link them
                if entity.recipe_name.lower() in chunk.text.lower():
                    chunk.recipe_name = entity.recipe_name
                    break

    def _collect_image_metadata(
        self,
        pages: list[PageContent],
        pdf_name: str,
        recipe_id_map: dict[str, str],
    ) -> list[ImageMetadata]:
        """Collect image metadata from extracted pages."""
        images: list[ImageMetadata] = []

        for page in pages:
            for img_path in page.image_paths:
                images.append(ImageMetadata(
                    image_path=img_path,
                    source_pdf=pdf_name,
                    page_number=page.page_number,
                ))

        return images


# ================================================================
# CLI Entry Point
# ================================================================

def main():
    """CLI entry point for running the ingestion pipeline."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    pipeline = IngestionPipeline()
    pipeline.setup()

    if len(sys.argv) > 1:
        target = sys.argv[1]
        if Path(target).is_file():
            pipeline.ingest(target)
        elif Path(target).is_dir():
            pipeline.ingest_directory(target)
        else:
            print(f"Error: {target} is not a valid file or directory")
            sys.exit(1)
    else:
        pipeline.ingest_directory(settings.pdf_input_dir)


if __name__ == "__main__":
    main()
