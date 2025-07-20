"""
API route handlers for the RAG system.

Endpoints:
  POST /api/query     — Submit a question, get a cited response
  GET  /api/recipes   — List all recipes in the knowledge graph
  GET  /api/health    — Health check
  GET  /api/images/{filename} — Serve recipe images
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.config import get_settings
from backend.models import QueryRequest, QueryResponse, RecipeSummary
from backend.retrieval.hybrid import HybridRetriever
from backend.generation.synthesizer import ResponseSynthesizer
from backend.ingestion.graph_builder import GraphBuilder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["RAG API"])

# Lazy-initialized singletons (created on first request)
_retriever: HybridRetriever | None = None
_synthesizer: ResponseSynthesizer | None = None
_graph: GraphBuilder | None = None


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_synthesizer() -> ResponseSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = ResponseSynthesizer()
    return _synthesizer


def _get_graph() -> GraphBuilder:
    global _graph
    if _graph is None:
        _graph = GraphBuilder()
    return _graph


# ================================================================
# Endpoints
# ================================================================

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Submit a natural language query and receive a cited response.

    The system will:
    1. Route the query (graph/vector/hybrid)
    2. Retrieve relevant data from Neo4j and/or Qdrant
    3. Synthesize a response with inline citations

    Example request:
    ```json
    {
        "question": "Find me a spicy Japanese recipe without pork",
        "include_images": true,
        "top_k": 5
    }
    ```
    """
    logger.info(f"Query received: {request.question[:100]}...")

    try:
        retriever = _get_retriever()
        synthesizer = _get_synthesizer()

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

        logger.info(
            f"Response generated: {len(response.citations)} citations, "
            f"type={response.query_type}"
        )
        return response

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.get("/recipes", response_model=list[RecipeSummary])
async def list_recipes() -> list[RecipeSummary]:
    """List all recipes in the knowledge graph."""
    try:
        graph = _get_graph()
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
        raise HTTPException(status_code=500, detail="Failed to retrieve recipes")


@router.get("/health")
async def health_check():
    """System health check."""
    return {
        "status": "healthy",
        "service": "domain-specific-multimodal-rag",
        "components": {
            "api": "up",
            "neo4j": _check_neo4j(),
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

def _check_neo4j() -> str:
    try:
        graph = _get_graph()
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
