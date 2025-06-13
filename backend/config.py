"""
Centralized configuration management using Pydantic Settings.
All environment variables are loaded from .env and validated at startup.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- OpenAI ---
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL"
    )

    # --- Neo4j ---
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="ragpassword", env="NEO4J_PASSWORD")

    # --- Qdrant ---
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_text_collection: str = Field(
        default="recipe_text", env="QDRANT_TEXT_COLLECTION"
    )
    qdrant_image_collection: str = Field(
        default="recipe_images", env="QDRANT_IMAGE_COLLECTION"
    )

    # --- Embedding Models ---
    text_embedding_model: str = Field(
        default="BAAI/bge-m3", env="TEXT_EMBEDDING_MODEL"
    )
    clip_model: str = Field(default="ViT-B-32", env="CLIP_MODEL")
    clip_pretrained: str = Field(
        default="laion2b_s34b_b79k", env="CLIP_PRETRAINED"
    )

    # --- Ingestion ---
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    pdf_input_dir: str = Field(default="data/raw", env="PDF_INPUT_DIR")
    image_output_dir: str = Field(default="data/images", env="IMAGE_OUTPUT_DIR")

    # --- API ---
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    cors_origins: str = Field(
        default="http://localhost:5173", env="CORS_ORIGINS"
    )

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton. Call this instead of instantiating directly."""
    return Settings()
