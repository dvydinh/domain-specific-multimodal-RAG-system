"""
Ingestion pipeline orchestrator with crash-resilient checkpointing.

Coordinates the full ETL flow:
  PDF → Extract → Chunk → Entity Extract → Graph Build → Vector Embed

Features:
  - Checkpoint/Resume: After each page is processed, progress is saved to
    disk. If the process crashes, it resumes from the last completed page.
  - Saga-protected inserts: Vector storage failures trigger compensating
    rollbacks in the Graph DB via SagaTransactionManager.
  - Batched entity extraction: Chunks are grouped to reduce LLM API calls.
"""

import logging
import re
import asyncio
import json
from pathlib import Path
from typing import Optional

from backend.config import get_settings
from backend.models import ChunkMetadata, ImageMetadata
from backend.ingestion.extractor import PDFExtractor, PageContent
from backend.ingestion.chunker import TextChunker
from backend.ingestion.entity_extractor import EntityExtractor
from backend.ingestion.graph_builder import GraphBuilder
from backend.ingestion.vector_store import VectorStoreManager
from backend.ingestion.saga import SagaTransactionManager

logger = logging.getLogger(__name__)

_CHECKPOINT_DIR = Path("data")
_CHECKPOINT_FILE = _CHECKPOINT_DIR / ".ingestion_checkpoint.json"


class IngestionPipeline:
    """
    End-to-end ingestion pipeline for recipe PDFs.
    Supports checkpoint/resume for crash resilience.
    """

    def __init__(
        self,
        extractor: Optional[PDFExtractor] = None,
        chunker: Optional[TextChunker] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        graph_builder: Optional[GraphBuilder] = None,
        vector_store: Optional[VectorStoreManager] = None,
        saga_manager: Optional[SagaTransactionManager] = None,
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
        self.saga_manager = saga_manager or SagaTransactionManager()

    def setup(self):
        """Initialize database schemas."""
        self.graph_builder.create_constraints()
        self.vector_store.create_collections()

    # ================================================================
    # Checkpoint helpers
    # ================================================================

    def _load_checkpoint(self, pdf_path: str) -> int:
        """Load last completed page index for this PDF. Returns -1 if none."""
        try:
            if _CHECKPOINT_FILE.exists():
                data = json.loads(_CHECKPOINT_FILE.read_text())
                if data.get("pdf") == pdf_path:
                    last_page = data.get("last_page", -1)
                    logger.info(f"Resuming from checkpoint: page {last_page + 1}")
                    return last_page
        except (json.JSONDecodeError, IOError):
            pass
        return -1

    def _save_checkpoint(self, pdf_path: str, page_index: int) -> None:
        """Save progress after each page is fully processed."""
        _CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        _CHECKPOINT_FILE.write_text(json.dumps({
            "pdf": pdf_path,
            "last_page": page_index,
        }))

    def _clear_checkpoint(self) -> None:
        """Remove checkpoint file after successful completion."""
        try:
            _CHECKPOINT_FILE.unlink(missing_ok=True)
        except IOError:
            pass

    # ================================================================
    # Main ingestion flow
    # ================================================================

    async def aingest(self, pdf_path: str) -> dict:
        """Process a single PDF with checkpoint/resume support."""
        pdf_name = Path(pdf_path).name
        logger.info(f"Starting ingestion: {pdf_name}")

        stats = {
            "pdf": pdf_name,
            "pages_extracted": 0,
            "chunks_created": 0,
            "entities_extracted": 0,
            "recipes_added": 0,
            "text_vectors": 0,
            "image_vectors": 0,
        }

        # Load checkpoint — skip already-processed pages
        last_completed_page = self._load_checkpoint(pdf_path)

        pages_generator = self.extractor.extract(pdf_path)
        current_recipe = None

        for page_index, page in enumerate(pages_generator):
            # Skip pages that were already processed before a crash
            if page_index <= last_completed_page:
                logger.info(f"  Skipping page {page_index} (already checkpointed)")
                continue

            stats["pages_extracted"] += 1

            # Step 2: Chunk (CPU-bound)
            page_chunks = await asyncio.to_thread(self._chunk_pages, [page], pdf_name)
            stats["chunks_created"] += len(page_chunks)

            if not page_chunks and not page.image_paths:
                self._save_checkpoint(pdf_path, page_index)
                continue

            # Step 3: Extract Entities (batched LLM calls)
            chunk_texts = [c.text for c in page_chunks]
            entities = []
            if chunk_texts:
                entities = await self.entity_extractor.aextract_batch(chunk_texts)
                stats["entities_extracted"] += len(entities)

            # Step 4: Build Graph
            recipe_id_map = {}
            if entities:
                recipe_id_map = await asyncio.to_thread(
                    self.graph_builder.add_recipes, entities, pdf_name
                )
                stats["recipes_added"] += len(recipe_id_map)

            # Assign recipe names to chunks
            for chunk in page_chunks:
                for entity in entities:
                    if re.search(rf"\b{re.escape(entity.recipe_name.lower())}\b", chunk.text.lower()):
                        current_recipe = entity.recipe_name
                        break

            # Step 5: Embed and Store (Saga-protected)
            async def _phase_2_insert(recipe_id_map, page_chunks):
                if page_chunks:
                    await asyncio.to_thread(
                        self.vector_store.embed_and_store_chunks, page_chunks, recipe_id_map
                    )
                if page.image_paths:
                    image_metadata = self._collect_image_metadata([page], pdf_name, recipe_id_map, page_chunks)
                    await asyncio.to_thread(
                        self.vector_store.embed_and_store_images, image_metadata, recipe_id_map
                    )

            async def _phase_2_rollback(recipe_id_map, page_chunks):
                if recipe_id_map:
                    await asyncio.to_thread(
                        self.graph_builder.delete_recipes, list(recipe_id_map.values())
                    )

            await self.saga_manager.execute_insert(
                insert_fn=_phase_2_insert,
                rollback_fn=_phase_2_rollback,
                recipe_id_map=recipe_id_map,
                page_chunks=page_chunks,
            )

            stats["text_vectors"] += len(page_chunks)
            stats["image_vectors"] += len(page.image_paths) if page.image_paths else 0

            # Checkpoint: this page is fully done
            self._save_checkpoint(pdf_path, page_index)

        # All pages processed — clear checkpoint
        self._clear_checkpoint()
        logger.info(f"Ingestion complete: {stats}")
        return stats

    async def aingest_directory(self, directory: str) -> list[dict]:
        """Process all PDFs with per-file checkpointing."""
        dir_path = Path(directory)
        pdf_files = sorted(dir_path.glob("*.pdf"))
        all_stats = []

        for pdf_path in pdf_files:
            try:
                stats = await self.aingest(str(pdf_path))
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Failed {pdf_path.name}: {e}")
                all_stats.append({"pdf": pdf_path.name, "error": str(e)})

        return all_stats

    def _chunk_pages(self, pages, pdf_name):
        return self.chunker.chunk_pages(pages, source_pdf=pdf_name)

    def _collect_image_metadata(self, pages, pdf_name, recipe_id_map, chunks):
        images = []
        page_to_recipes = {}
        for chunk in chunks:
            if chunk.recipe_name and chunk.recipe_name in recipe_id_map:
                page_to_recipes.setdefault(chunk.page_number, set()).add(chunk.recipe_name)
        for page in pages:
            page_recipes = list(page_to_recipes.get(page.page_number, set()))
            recipe_name = page_recipes[0] if page_recipes else None
            for img_path in page.image_paths:
                images.append(ImageMetadata(
                    image_path=img_path, source_pdf=pdf_name,
                    page_number=page.page_number, recipe_name=recipe_name,
                ))
        return images


if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m backend.ingestion.pipeline <pdf_path_or_directory>")
        sys.exit(1)

    path = sys.argv[1]
    pipeline = IngestionPipeline()
    pipeline.setup()

    if os.path.isdir(path):
        asyncio.run(pipeline.aingest_directory(path))
    else:
        asyncio.run(pipeline.aingest(path))
