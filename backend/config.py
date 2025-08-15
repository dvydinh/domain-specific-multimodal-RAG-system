"""
Centralized configuration management using Pydantic Settings.
All environment variables are loaded from .env and validated at startup.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, ConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- OpenAI ---
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # --- Neo4j ---
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_user: str = Field(default="neo4j")
    neo4j_password: str = Field(default="ragpassword")

    # --- Qdrant ---
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_text_collection: str = Field(default="recipe_text")
    qdrant_image_collection: str = Field(default="recipe_images")

    # --- Embedding Models ---
    text_embedding_model: str = Field(default="BAAI/bge-m3")
    clip_model: str = Field(default="ViT-B-32")
    clip_pretrained: str = Field(default="laion2b_s34b_b79k")

    # --- Ingestion ---
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    pdf_input_dir: str = Field(default="data/raw")
    image_output_dir: str = Field(default="data/images")

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    cors_origins: str = Field(default="http://localhost:5173")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton. Call this instead of instantiating directly."""
    return Settings()
