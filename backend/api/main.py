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
    """
    Application lifespan: Manage database connections and singletons.
    Ensures clean startup and teardown to prevent connection leaks.
    """
    logger.info("=" * 60)
    logger.info("Initializing Production RAG Services")
    logger.info("=" * 60)

    from backend.retrieval.hybrid import HybridRetriever
    from backend.generation.synthesizer import ResponseSynthesizer
    from backend.ingestion.graph_builder import GraphBuilder
    from backend.ingestion.vector_store import VectorStoreManager

    # Initialize singletons on app.state for Dependency Injection
    app.state.retriever = HybridRetriever()
    app.state.synthesizer = ResponseSynthesizer()
    app.state.graph = GraphBuilder()
    app.state.vector_store = VectorStoreManager()

    yield

    logger.info("Shutting down RAG services...")
    app.state.retriever.close()
    app.state.graph.close()
    # VectorStoreManager doesn't have a close(), but we manage its lifecycle here
    logger.info("Shutdown complete.")


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
