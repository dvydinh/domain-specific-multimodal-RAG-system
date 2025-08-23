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

        # === Stream Pages to prevent OOM ===
        logger.info("Extracting and processing streaming pages...")
        pages_generator = self.extractor.extract(pdf_path)
        current_recipe = None

        for page in pages_generator:
            stats["pages_extracted"] += 1
            logger.info(f"  → Processing page {stats['pages_extracted']}...")

            # === Step 2: Chunk text ===
            page_chunks = self._chunk_pages([page], pdf_name)
            stats["chunks_created"] += len(page_chunks)

            if not page_chunks and not page.image_paths:
                continue

            # === Step 3: Extract entities via LLM ===
            chunk_texts = [c.text for c in page_chunks]
            entities = []
            if chunk_texts:
                entities = self.entity_extractor.extract_batch(chunk_texts)
                stats["entities_extracted"] += len(entities)

            # === Step 4: Build knowledge graph ===
            recipe_id_map = {}
            if entities:
                recipe_id_map = self.graph_builder.add_recipes(
                    entities, source_pdf=pdf_name
                )
                stats["recipes_added"] += len(recipe_id_map)

            # Assign recipe names to chunks for cross-referencing continuously
            for chunk in page_chunks:
                for entity in entities:
                    if entity.recipe_name.lower() in chunk.text.lower():
                        current_recipe = entity.recipe_name
                        break
                if current_recipe:
                    chunk.recipe_name = current_recipe

            # === Step 5: Embed and store vectors ===
            if page_chunks:
                text_count = self.vector_store.embed_and_store_chunks(
                    page_chunks, recipe_id_map
                )
                stats["text_vectors"] += text_count

            if page.image_paths:
                image_metadata = self._collect_image_metadata(
                    [page], pdf_name, recipe_id_map, page_chunks
                )
                image_count = self.vector_store.embed_and_store_images(
                    image_metadata, recipe_id_map
                )
                stats["image_vectors"] += image_count

        if stats["pages_extracted"] == 0:
            logger.warning(f"No content extracted from {pdf_name}")
            return stats

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
        """Convert extracted pages into text chunks, preserving blocks for bbox."""
        return self.chunker.chunk_pages(pages, source_pdf=pdf_name)

    def _collect_image_metadata(
        self,
        pages: list[PageContent],
        pdf_name: str,
        recipe_id_map: dict[str, str],
        chunks: list[ChunkMetadata],
    ) -> list[ImageMetadata]:
        """Collect image metadata from extracted pages and map to recipes via chunks."""
        images: list[ImageMetadata] = []

        # Build mapping from page_number to a set of recipe_names found on that page
        page_to_recipes = {}
        for chunk in chunks:
            if chunk.recipe_name and chunk.recipe_name in recipe_id_map:
                page_to_recipes.setdefault(chunk.page_number, set()).add(chunk.recipe_name)

        for page in pages:
            # Best-effort: map images to the dominant recipe on the page
            page_recipes = list(page_to_recipes.get(page.page_number, set()))
            recipe_name = page_recipes[0] if page_recipes else None

            for img_path in page.image_paths:
                images.append(ImageMetadata(
                    image_path=img_path,
                    source_pdf=pdf_name,
                    page_number=page.page_number,
                    recipe_name=recipe_name,
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
            logger.error(f"Invalid target: {target} is not a valid file or directory")
            sys.exit(1)
    else:
        pipeline.ingest_directory(settings.pdf_input_dir)


if __name__ == "__main__":
    main()
