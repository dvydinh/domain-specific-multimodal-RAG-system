# Domain-Specific Multimodal RAG System

A hybrid Retrieval-Augmented Generation system combining **Constraint-based Knowledge Graph Filtering** (Neo4j) and **Vector Database** (Qdrant) with **Reciprocal Rank Fusion (RRF)** for accurate, hallucination-resistant multimodal querying over a cooking/nutrition corpus. Note: This implements Constraint-based Graph Filtering and is distinct from community-summarization approaches like Microsoft's GraphRAG.

## Architecture

```
User Query
    │
    ▼
┌───────────────┐
│  LLM Router   │   Classify → GRAPH / VECTOR / HYBRID
└───────┬───────┘
        │
        ▼ (constraints detected)
┌───────────────┐
│ Neo4j Graph   │   NL → Cypher → Exact filtering (tags, ingredients, exclusions)
│ Retrieval     │
└───────┬───────┘
        │ filtered recipe IDs
        ▼
┌───────────────┐
│ Qdrant Vector │   Semantic search on pre-filtered scope → text chunks + images
│ Retrieval     │
└───────┬───────┘
        │
        ▼
┌───────────────┐
│ LLM Synthesis │   Graph + Vector contexts → Answer with [citations]
└───────────────┘
```

### Why Hybrid?

| Problem | Pure Vector RAG | Our Hybrid System |
|---------|----------------|-------------------|
| "Recipes **without** pork" | Returns pork recipes (embedding similarity) | Graph excludes pork via `WHERE NOT` → correct |
| "Japanese AND spicy AND vegan" | Inconsistent multi-constraint results | Graph applies all filters precisely |
| Hallucination | LLM may invent facts | Vector search scoped to graph-validated IDs only |
| Image retrieval | No structured link to recipes | Images linked via `neo4j_recipe_id` payload |

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Knowledge Graph | Neo4j 5.x | Structured recipe data, logical querying |
| Vector Database | Qdrant 1.10 | Semantic text search, image search (HNSW) |
| Text Embeddings | BAAI/bge-m3 | 1024-dim multilingual dense vectors |
| Image Embeddings | CLIP ViT-B/32 | 512-dim cross-modal embeddings |
| LLM | GPT-4o-mini | Entity extraction, routing, synthesis |
| Backend | FastAPI | REST API with CORS |
| Frontend | React 18 + Vite | Chat UI with citation popups |
| Orchestration | LangChain | LLM chaining and prompt management |

## Project Structure

```
.
├── backend/
│   ├── api/               # FastAPI application
│   │   ├── main.py        # App entry point, CORS, lifespan
│   │   └── routes.py      # Query, recipes, health endpoints
│   ├── generation/
│   │   └── synthesizer.py # LLM response generation with citations
│   ├── ingestion/
│   │   ├── chunker.py     # Text chunking with overlap
│   │   ├── entity_extractor.py  # LLM-based recipe entity extraction
│   │   ├── extractor.py   # PDF text + image extraction (PyMuPDF)
│   │   ├── graph_builder.py     # Neo4j graph construction
│   │   ├── pipeline.py    # ETL orchestrator
│   │   └── vector_store.py      # Qdrant collection management
│   ├── retrieval/
│   │   ├── graph_retriever.py   # NL → Cypher → Neo4j
│   │   ├── hybrid.py      # 3-step retrieval orchestrator
│   │   ├── router.py      # Query classification (graph/vector/hybrid)
│   │   └── vector_retriever.py  # Qdrant semantic search
│   ├── config.py          # Pydantic Settings
│   └── models.py          # Data models
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.jsx
│   │   │   ├── CitationPopup.jsx
│   │   │   └── MessageBubble.jsx
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   └── package.json
├── tests/
│   ├── test_chunker.py
│   ├── test_models.py
│   └── test_router.py
├── data/
│   └── sample/recipes.json
├── docs/
│   ├── architecture_design.md
│   └── literature_review.md
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose

### 1. Infrastructure

```bash
# Start Neo4j and Qdrant
docker-compose up -d

# Verify services
curl http://localhost:6333/healthz   # Qdrant
open http://localhost:7474           # Neo4j Browser
```

### 2. Backend

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key

# Run ingestion (process PDFs)
python -m backend.ingestion.pipeline data/raw/

# Start API server
uvicorn backend.api.main:app --reload
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## Usage

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/query` | Submit a question, get a cited response |
| `GET` | `/api/recipes` | List all recipes in the knowledge graph |
| `GET` | `/api/health` | System health check |
| `GET` | `/api/images/{filename}` | Serve recipe images |

### Query Example

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Find a spicy Japanese recipe with pork but without scallion",
    "include_images": true,
    "top_k": 5
  }'
```

Response:
```json
{
  "response": "Based on your criteria, here is a Japanese recipe... [1]",
  "citations": {
    "1": {
      "id": "1",
      "text": "Recipe: Tonkotsu Ramen | Ingredients: pork bone, soy sauce...",
      "recipe_name": "Tonkotsu Ramen",
      "image_url": "/api/images/ramen_p1_img0.jpg"
    }
  },
  "query_type": "hybrid",
  "graph_results_count": 2,
  "vector_results_count": 5
}
```

## Key Design Decisions

### Graph-First Filtering

The system routes constraint-heavy queries through Neo4j **first**, collecting a set of valid recipe IDs. These IDs are then used as a Qdrant payload filter, constraining the vector search space. This eliminates false positives from embedding similarity (e.g., "without pork" matching pork recipes).

### HNSW Tuning & Memory Math

Qdrant collections use `m=16, ef_construct=100` for the HNSW index. This provides strong recall (>95%) while keeping search latency under 10ms for collections up to 100K vectors.

**Physical RAM required for 50 million text vectors (BGE-M3 dims=1024):**
Memory = N_vectors * (dim * 4 + M * 4) * 1.5 
`RAM = 50,000,000 * (1024 * 4 + 16 * 4) * 1.5 = ~312 GB`

### LLM Entity Extraction

Rather than manually parsing recipes, we use GPT-4o-mini with function calling to extract structured JSON (recipe name, ingredients, tags) from raw text. This handles varied formatting across different cookbooks.

## Testing

```bash
# Run unit tests
python -m pytest tests/ -v

# Tests cover:
# - Text chunking (overlap, boundaries, edge cases)
# - Pydantic model validation
# - Query router heuristics
```

## License

MIT
