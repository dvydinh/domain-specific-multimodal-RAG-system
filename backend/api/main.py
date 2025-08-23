"""
FastAPI application entry point.

Configures CORS, lifespan events, and mounts the API routes.
Run with: uvicorn backend.api.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from backend.config import get_settings
from backend.api.routes import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info("=" * 60)
    logger.info("Starting Domain-Specific Multimodal RAG System")
    logger.info("=" * 60)

    settings = get_settings()
    logger.info(f"LLM model: {settings.google_model}")
    logger.info(f"Neo4j: {settings.neo4j_uri}")
    logger.info(f"Qdrant: {settings.qdrant_host}:{settings.qdrant_port}")

    # Ensure image directory exists
    Path(settings.image_output_dir).mkdir(parents=True, exist_ok=True)

    yield

    logger.info("Shutting down RAG system")


# Create the FastAPI app
app = FastAPI(
    title="Domain-Specific Multimodal RAG System",
    description=(
        "A hybrid RAG system combining Neo4j Knowledge Graph and "
        "Qdrant Vector Database for multimodal recipe querying. "
        "Features graph-filtered vector search to eliminate hallucination."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router)

# Mount static files for images (if directory exists)
image_dir = Path(settings.image_output_dir)
if image_dir.exists():
    app.mount(
        "/static/images",
        StaticFiles(directory=str(image_dir)),
        name="images",
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Domain-Specific Multimodal RAG System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
