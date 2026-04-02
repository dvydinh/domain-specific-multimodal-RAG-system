"""
API route handlers for the RAG system.

Endpoints:
  POST /api/query     — Submit a question, get a cited response
  GET  /api/recipes   — List all recipes in the knowledge graph
  GET  /api/health    — Health check
  GET  /api/images/{filename} — Serve recipe images
"""

import logging
import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Depends, Request
from fastapi.responses import FileResponse

from backend.config import get_settings
from backend.models import QueryRequest, QueryResponse, RecipeSummary
from backend.retrieval.hybrid import HybridRetriever
from backend.generation.synthesizer import ResponseSynthesizer
from backend.ingestion.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG API"])

# ================================================================
# Dependencies (DI for Scalability)
# ================================================================

def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever

def get_synthesizer(request: Request) -> ResponseSynthesizer:
    return request.app.state.synthesizer

def get_graph(request: Request) -> GraphBuilder:
    return request.app.state.graph

def get_vector_store(request: Request) -> VectorStoreManager:
    return request.app.state.vector_store


# ================================================================
# Endpoints
# ================================================================

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    synthesizer: ResponseSynthesizer = Depends(get_synthesizer)
) -> QueryResponse:
    """
    Submit a natural language query and receive a cited response.
    """
    logger.info(f"Query received: {request.question[:100]}...")

    try:
        # Step 1-3: Hybrid retrieval (Async)
        retrieval_results = await retriever.aretrieve(
            query=request.question,
            top_k=request.top_k,
            include_images=request.include_images,
        )

        # Step 4: LLM synthesis with citations (Async)
        response = await synthesizer.asynthesize(
            query=request.question,
            retrieval_results=retrieval_results,
        )
        return response

    except Exception as e:
        # Security: Log internal error privately, mask for Client
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred while processing your query. Please try again later."
        )


@router.get("/recipes", response_model=list[RecipeSummary])
async def list_recipes(graph: GraphBuilder = Depends(get_graph)) -> list[RecipeSummary]:
    """List all recipes in the knowledge graph."""
    try:
        recipes = graph.get_all_recipes()

        return [
            RecipeSummary(
                id=r.get("id", ""),
                name=r.get("name", ""),
                cuisine=r.get("cuisine"),
                tags=r.get("tags", []),
                ingredient_count=r.get("ingredient_count", 0),
            )
            for r in recipes
        ]
    except Exception as e:
        logger.error(f"Failed to list recipes: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recipe list.")


@router.get("/health")
async def health_check(request: Request):
    """System health check (Live monitoring)."""
    return {
        "status": "healthy",
        "service": "domain-specific-multimodal-rag",
        "components": {
            "api": "up",
            "neo4j": _check_neo4j(request.app.state.graph),
            "qdrant": _check_qdrant(),
        }
    }


@router.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve a recipe image from the data directory."""
    settings = get_settings()
    image_path = Path(settings.image_output_dir) / filename

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(image_path))


# ================================================================
# Health Check Helpers
# ================================================================

def _check_neo4j(graph: GraphBuilder) -> str:
    try:
        graph.get_all_recipes()
        return "up"
    except Exception:
        return "down"


def _check_qdrant() -> str:
    try:
        from qdrant_client import QdrantClient
        settings = get_settings()
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        client.get_collections()
        return "up"
    except Exception:
        return "down"


# ================================================================
# Document Upload
# ================================================================

from backend.ingestion.pipeline import IngestionPipeline

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    graph: GraphBuilder = Depends(get_graph),
    vector_store: VectorStoreManager = Depends(get_vector_store)
):
    """
    Upload a PDF cookbook document for ingestion.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    # Save file safely (using to_thread as previously hardened)
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / file.filename

    try:
        with open(dest, "wb") as buf:
            await asyncio.to_thread(shutil.copyfileobj, file.file, buf)
    except Exception as e:
        logger.error(f"Failed to save file {dest}: {e}")
        raise HTTPException(status_code=500, detail="Failed to securely save the uploaded document.")

    # PRODUCTION FIX: Initialize Pipeline with shared DB connections
    pipeline = IngestionPipeline(graph_builder=graph, vector_store=vector_store)
    
    # Native FastAPI Async Background Task (No manual event loop needed!)
    background_tasks.add_task(pipeline.aingest, str(dest))

    return {
        "status": "accepted",
        "filename": file.filename,
        "message": "File uploaded successfully. Ingestion pipeline started in background."
    }

