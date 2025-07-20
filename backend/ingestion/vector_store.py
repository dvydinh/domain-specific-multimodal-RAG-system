"""
Vector store manager for Qdrant.

Manages two collections:
  - recipe_text: BGE-M3 text embeddings (dim=1024)
  - recipe_images: CLIP image embeddings (dim=512)

Each vector carries a payload with neo4j_recipe_id for cross-referencing
with the knowledge graph.
"""

import logging
from typing import Optional
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from backend.config import get_settings
from backend.models import ChunkMetadata, ImageMetadata

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages Qdrant vector collections for text and image embeddings.

    Handles collection creation, embedding generation, and vector upsert
    with metadata payloads linked to Neo4j recipe IDs.
    """

    # Embedding dimensions
    TEXT_DIM = 1024   # BGE-M3
    IMAGE_DIM = 512   # CLIP ViT-B/32

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ):
        settings = get_settings()
        self.client = QdrantClient(
            host=host or settings.qdrant_host,
            port=port or settings.qdrant_port,
        )
        self.text_collection = settings.qdrant_text_collection
        self.image_collection = settings.qdrant_image_collection

        # Lazy-loaded embedding models
        self._text_model: Optional[SentenceTransformer] = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None

        logger.info(
            f"Connected to Qdrant at {host or settings.qdrant_host}:{port or settings.qdrant_port}"
        )

    # ================================================================
    # Collection Setup
    # ================================================================

    def create_collections(self):
        """Create Qdrant collections with HNSW configuration if they don't exist."""
        hnsw_config = qmodels.HnswConfigDiff(
            m=16,
            ef_construct=100,
        )

        # Text collection
        if not self._collection_exists(self.text_collection):
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config=qmodels.VectorParams(
                    size=self.TEXT_DIM,
                    distance=qmodels.Distance.COSINE,
                    hnsw_config=hnsw_config,
                ),
            )
            logger.info(f"Created collection '{self.text_collection}' (dim={self.TEXT_DIM})")

        # Image collection
        if not self._collection_exists(self.image_collection):
            self.client.create_collection(
                collection_name=self.image_collection,
                vectors_config=qmodels.VectorParams(
                    size=self.IMAGE_DIM,
                    distance=qmodels.Distance.COSINE,
                    hnsw_config=hnsw_config,
                ),
            )
            logger.info(f"Created collection '{self.image_collection}' (dim={self.IMAGE_DIM})")

    def _collection_exists(self, name: str) -> bool:
        """Check if a collection already exists."""
        try:
            self.client.get_collection(name)
            return True
        except Exception:
            return False

    # ================================================================
    # Embedding Models (lazy loading)
    # ================================================================

    def _get_text_model(self) -> SentenceTransformer:
        """Lazily load the BGE-M3 text embedding model."""
        if self._text_model is None:
            settings = get_settings()
            logger.info(f"Loading text embedding model: {settings.text_embedding_model}")
            self._text_model = SentenceTransformer(settings.text_embedding_model)
        return self._text_model

    def _get_clip_model(self):
        """Lazily load the CLIP model for image embeddings."""
        if self._clip_model is None:
            import open_clip
            settings = get_settings()

            logger.info(f"Loading CLIP model: {settings.clip_model}")
            model, _, preprocess = open_clip.create_model_and_transforms(
                settings.clip_model,
                pretrained=settings.clip_pretrained,
            )
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._clip_tokenizer = open_clip.get_tokenizer(settings.clip_model)

        return self._clip_model, self._clip_preprocess, self._clip_tokenizer

    # ================================================================
    # Text Embedding & Upsert
    # ================================================================

    def embed_and_store_chunks(
        self,
        chunks: list[ChunkMetadata],
        recipe_id_map: dict[str, str],
    ) -> int:
        """
        Embed text chunks and store them in Qdrant.

        Args:
            chunks: Text chunks with metadata.
            recipe_id_map: Mapping of recipe_name -> neo4j_recipe_id.

        Returns:
            Number of vectors stored.
        """
        if not chunks:
            return 0

        model = self._get_text_model()
        texts = [chunk.text for chunk in chunks]

        logger.info(f"Embedding {len(texts)} text chunks...")
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Match chunk to recipe ID via recipe name
            recipe_id = None
            if chunk.recipe_name:
                recipe_id = recipe_id_map.get(chunk.recipe_name)

            point = qmodels.PointStruct(
                id=i,
                vector=embedding.tolist(),
                payload={
                    "text": chunk.text,
                    "neo4j_recipe_id": recipe_id or "",
                    "recipe_name": chunk.recipe_name or "",
                    "source_pdf": chunk.source_pdf,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "bbox": chunk.bbox,
                },
            )
            points.append(point)

        # Batch upsert (Qdrant handles batching internally)
        self.client.upsert(
            collection_name=self.text_collection,
            points=points,
        )

        logger.info(f"Stored {len(points)} text vectors in '{self.text_collection}'")
        return len(points)

    # ================================================================
    # Image Embedding & Upsert
    # ================================================================

    def embed_and_store_images(
        self,
        image_metadata: list[ImageMetadata],
        recipe_id_map: dict[str, str],
    ) -> int:
        """
        Embed images using CLIP and store them in Qdrant.

        Args:
            image_metadata: Image metadata with file paths.
            recipe_id_map: Mapping of recipe_name -> neo4j_recipe_id.

        Returns:
            Number of image vectors stored.
        """
        if not image_metadata:
            return 0

        import torch
        from PIL import Image

        model, preprocess, _ = self._get_clip_model()
        points = []

        for idx, img_meta in enumerate(image_metadata):
            img_path = Path(img_meta.image_path)
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0)

                with torch.no_grad():
                    embedding = model.encode_image(image_tensor)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                recipe_id = ""
                if img_meta.recipe_name:
                    recipe_id = recipe_id_map.get(img_meta.recipe_name, "")

                point = qmodels.PointStruct(
                    id=idx,
                    vector=embedding.squeeze().tolist(),
                    payload={
                        "neo4j_recipe_id": recipe_id,
                        "recipe_name": img_meta.recipe_name or "",
                        "image_path": str(img_meta.image_path),
                        "source_pdf": img_meta.source_pdf,
                        "page_number": img_meta.page_number,
                    },
                )
                points.append(point)

            except Exception as e:
                logger.error(f"Failed to embed image {img_path}: {e}")

        if points:
            self.client.upsert(
                collection_name=self.image_collection,
                points=points,
            )

        logger.info(f"Stored {len(points)} image vectors in '{self.image_collection}'")
        return len(points)

    # ================================================================
    # Search
    # ================================================================

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        recipe_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Semantic search over text chunks.

        Args:
            query: Natural language query.
            top_k: Number of results.
            recipe_ids: Optional list of neo4j_recipe_ids to filter by.

        Returns:
            List of search results with text and metadata.
        """
        model = self._get_text_model()
        query_vector = model.encode(query, normalize_embeddings=True).tolist()

        search_filter = None
        if recipe_ids:
            search_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="neo4j_recipe_id",
                        match=qmodels.MatchAny(any=recipe_ids),
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.text_collection,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
        )

        return [
            {
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "recipe_name": hit.payload.get("recipe_name", ""),
                "neo4j_recipe_id": hit.payload.get("neo4j_recipe_id", ""),
                "source_pdf": hit.payload.get("source_pdf", ""),
                "page_number": hit.payload.get("page_number", 0),
                "bbox": hit.payload.get("bbox", None),
            }
            for hit in results
        ]

    def search_images(
        self,
        query: str,
        top_k: int = 3,
        recipe_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Search for recipe images using text-to-image CLIP similarity.

        Args:
            query: Natural language description.
            top_k: Number of results.
            recipe_ids: Optional filter by recipe IDs.

        Returns:
            List of image results with paths and metadata.
        """
        import torch

        model, _, tokenizer = self._get_clip_model()
        text_tokens = tokenizer([query])

        with torch.no_grad():
            text_embedding = model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        query_vector = text_embedding.squeeze().tolist()

        search_filter = None
        if recipe_ids:
            search_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="neo4j_recipe_id",
                        match=qmodels.MatchAny(any=recipe_ids),
                    )
                ]
            )

        results = self.client.search(
            collection_name=self.image_collection,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=top_k,
        )

        return [
            {
                "score": hit.score,
                "image_path": hit.payload.get("image_path", ""),
                "recipe_name": hit.payload.get("recipe_name", ""),
                "neo4j_recipe_id": hit.payload.get("neo4j_recipe_id", ""),
            }
            for hit in results
        ]
