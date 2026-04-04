"""
Vector Store Manager for Qdrant.

Handles embedding and storage of text chunks (via fastembed/BGE)
and images (via OpenCLIP ViT-B/32) into separate Qdrant collections.

Supports filtered search by recipe IDs from the graph retriever,
enabling graph-constrained vector retrieval.
"""

import logging
from typing import Optional
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from fastembed import TextEmbedding

from backend.config import get_settings
from backend.models import ChunkMetadata, ImageMetadata

logger = logging.getLogger(__name__)


# Dimension constants — must match the embedding models used
TEXT_DIM = 384    # BAAI/bge-small-en-v1.5
IMAGE_DIM = 512   # OpenCLIP ViT-B/32


class VectorStoreManager:
    """
    Manages Qdrant vector collections for text and image embeddings.

    Text embeddings use fastembed (BAAI/bge-small-en-v1.5, 384-dim).
    Image embeddings use OpenCLIP ViT-B/32 (512-dim).
    Both collections carry neo4j_recipe_id payloads for graph-vector linking.
    """

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

        # Initialize text embedding model eagerly
        logger.info(f"Initializing text embedding: {settings.text_embedding_model}")
        self._dense_model = TextEmbedding(model_name=settings.text_embedding_model)

        # CLIP model loaded lazily (only needed for image queries)
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
        """Initialize Qdrant collections if they don't exist."""
        # Text collection (dense vectors only)
        if not self._collection_exists(self.text_collection):
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config={
                    "dense": qmodels.VectorParams(
                        size=TEXT_DIM,
                        distance=qmodels.Distance.COSINE,
                    )
                },
                on_disk_payload=True,
            )
            logger.info(f"Created collection '{self.text_collection}' (dim={TEXT_DIM})")

        # Image collection (CLIP vectors)
        if not self._collection_exists(self.image_collection):
            self.client.create_collection(
                collection_name=self.image_collection,
                vectors_config=qmodels.VectorParams(
                    size=IMAGE_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
                on_disk_payload=True,
            )
            logger.info(f"Created collection '{self.image_collection}' (dim={IMAGE_DIM})")

    def _collection_exists(self, name: str) -> bool:
        try:
            self.client.get_collection(name)
            return True
        except Exception:
            return False

    # ================================================================
    # Embedding Models
    # ================================================================

    @property
    def dense_model(self) -> TextEmbedding:
        return self._dense_model

    def _get_clip_model(self) -> tuple:
        """Lazy-load OpenCLIP model (only needed for image operations)."""
        if self._clip_model is None:
            import open_clip
            settings = get_settings()
            model, _, preprocess = open_clip.create_model_and_transforms(
                settings.clip_model,
                pretrained=settings.clip_pretrained,
            )
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._clip_tokenizer = open_clip.get_tokenizer(settings.clip_model)
            logger.info(f"Loaded CLIP model: {settings.clip_model}")
        return self._clip_model, self._clip_preprocess, self._clip_tokenizer

    # ================================================================
    # Ingestion: Text Chunks
    # ================================================================

    def embed_and_store_chunks(
        self,
        chunks: list[ChunkMetadata],
        recipe_id_map: dict[str, str],
    ) -> int:
        """
        Embed text chunks via BGE and upsert to Qdrant.

        Args:
            chunks: Text chunks with metadata.
            recipe_id_map: Mapping of recipe name → Neo4j UUID.

        Returns:
            Number of points upserted.
        """
        if not chunks:
            return 0

        # Filter out empty/whitespace-only chunks
        valid_chunks = [c for c in chunks if c.text and c.text.strip()]
        if not valid_chunks:
            logger.warning("No non-empty chunks to embed. Skipping batch.")
            return 0

        texts = [c.text.strip() for c in valid_chunks]

        # Embed using fastembed
        dense_vecs = list(self.dense_model.embed(texts))

        points = []
        for chunk, d_vec in zip(valid_chunks, dense_vecs):
            recipe_id = recipe_id_map.get(chunk.recipe_name, "") if chunk.recipe_name else ""

            point = qmodels.PointStruct(
                id=abs(hash(f"{chunk.source_pdf}_{chunk.page_number}_{chunk.chunk_index}")) % (10**15),
                vector={"dense": d_vec.tolist()},
                payload={
                    "text": chunk.text,
                    "neo4j_recipe_id": recipe_id,
                    "recipe_name": chunk.recipe_name or "",
                    "source_pdf": chunk.source_pdf,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "bbox": chunk.bbox,
                },
            )
            points.append(point)

        self.client.upsert(collection_name=self.text_collection, points=points)
        logger.info(f"Upserted {len(points)} text vectors to '{self.text_collection}'")
        return len(points)

    # ================================================================
    # Ingestion: Images
    # ================================================================

    def embed_and_store_images(
        self,
        image_metadata: list[ImageMetadata],
        recipe_id_map: dict[str, str],
    ) -> int:
        """Embed images via CLIP and upsert to Qdrant."""
        if not image_metadata:
            return 0

        import torch
        from PIL import Image

        model, preprocess, _ = self._get_clip_model()
        points = []

        for img_meta in image_metadata:
            img_path = Path(img_meta.image_path)
            if not img_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0)

                with torch.no_grad():
                    embedding = model.encode_image(image_tensor)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                recipe_id = recipe_id_map.get(img_meta.recipe_name, "") if img_meta.recipe_name else ""

                point = qmodels.PointStruct(
                    id=abs(hash(str(img_path))) % (10**15),
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
                logger.error(f"Failed to process image {img_path}: {e}")

        if points:
            self.client.upsert(collection_name=self.image_collection, points=points)
            logger.info(f"Upserted {len(points)} image vectors to '{self.image_collection}'")
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
            top_k: Number of results to return.
            recipe_ids: Optional list of Neo4j recipe IDs to filter by.

        Returns:
            List of result dicts with score, text, and metadata.
        """
        query_dense = list(self.dense_model.embed([query]))[0].tolist()

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

        results = self.client.query_points(
            collection_name=self.text_collection,
            query=query_dense,
            using="dense",
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
            }
            for hit in results.points
        ]

    def search_images(
        self,
        query: str,
        top_k: int = 3,
        recipe_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Text-to-image cross-modal search via CLIP."""
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

        results = self.client.query_points(
            collection_name=self.image_collection,
            query=query_vector,
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
            for hit in results.points
        ]
