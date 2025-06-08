# System Architecture Design

## Overview

A hybrid RAG system combining **Knowledge Graph** (Neo4j) and **Vector Database** (Qdrant)
for accurate, hallucination-resistant retrieval over a multimodal cooking/nutrition corpus.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        React Frontend                            │
│   ChatInterface → MessageBubble → CitationPopup (text + image)  │
└──────────────────────┬───────────────────────────────────────────┘
                       │ HTTP/JSON
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                     FastAPI Backend                               │
│                                                                  │
│  ┌─────────────┐   ┌──────────────────┐   ┌─────────────────┐  │
│  │ Query Router │──▶│ Hybrid Retriever │──▶│ LLM Synthesizer │  │
│  └─────────────┘   └────────┬─────────┘   └─────────────────┘  │
│                       ┌─────┴──────┐                             │
│                       ▼            ▼                              │
│               ┌───────────┐  ┌──────────┐                       │
│               │   Neo4j   │  │  Qdrant  │                       │
│               │  (Graph)  │  │ (Vector) │                       │
│               └───────────┘  └──────────┘                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Ingestion Pipeline

```
PDF Files
    │
    ▼
┌────────────────┐
│  PDF Extractor │── Extract text blocks + images
│  (PyMuPDF)     │
└───────┬────────┘
        │
   ┌────┴────┐
   ▼         ▼
┌──────┐  ┌──────────┐
│ Text │  │  Images  │
│Chunks│  │ (saved)  │
└──┬───┘  └────┬─────┘
   │           │
   ▼           │
┌────────────┐ │
│  LLM       │ │    Function Calling (GPT-4o-mini)
│  Entity    │ │    → Structured JSON output
│  Extractor │ │
└──┬─────────┘ │
   │           │
   ▼           │
┌────────────┐ │
│  Neo4j     │ │    Create nodes: Recipe, Ingredient, Tag
│  Graph     │ │    Create edges: CONTAINS_INGREDIENT, HAS_TAG
│  Builder   │ │
└──┬─────────┘ │
   │           │
   ▼           ▼
┌──────────────────┐
│  Vector Store    │    BGE-M3 for text, CLIP for images
│  (Qdrant)        │    Payload: neo4j_recipe_id, source_page
└──────────────────┘
```

---

## Neo4j Schema

### Node Labels

| Label | Properties | Description |
|-------|-----------|-------------|
| `Recipe` | `id` (UUID), `name`, `cuisine`, `source_pdf`, `page_number` | A single recipe |
| `Ingredient` | `name` (normalized lowercase) | A cooking ingredient |
| `Tag` | `name` (e.g., "Spicy", "Vegan", "Japanese") | Classification label |

### Relationship Types

| Relationship | From → To | Properties |
|-------------|-----------|------------|
| `CONTAINS_INGREDIENT` | Recipe → Ingredient | `quantity` (optional), `unit` (optional) |
| `HAS_TAG` | Recipe → Tag | — |

### Constraints & Indexes

```cypher
CREATE CONSTRAINT recipe_name_unique IF NOT EXISTS
FOR (r:Recipe) REQUIRE r.name IS UNIQUE;

CREATE CONSTRAINT ingredient_name_unique IF NOT EXISTS
FOR (i:Ingredient) REQUIRE i.name IS UNIQUE;

CREATE CONSTRAINT tag_name_unique IF NOT EXISTS
FOR (t:Tag) REQUIRE t.name IS UNIQUE;

CREATE INDEX recipe_id_index IF NOT EXISTS
FOR (r:Recipe) ON (r.id);
```

### Example Graph

```cypher
(:Recipe {name: "Tonkotsu Ramen", cuisine: "Japanese"})
    -[:CONTAINS_INGREDIENT]->(:Ingredient {name: "pork bone"})
    -[:CONTAINS_INGREDIENT]->(:Ingredient {name: "soy sauce"})
    -[:CONTAINS_INGREDIENT]->(:Ingredient {name: "scallion"})
    -[:HAS_TAG]->(:Tag {name: "Japanese"})
    -[:HAS_TAG]->(:Tag {name: "Spicy"})
```

---

## Qdrant Collection Design

### `recipe_text` Collection

| Field | Value |
|-------|-------|
| **Vector size** | 1024 (BGE-M3) |
| **Distance** | Cosine |
| **HNSW config** | `m=16`, `ef_construct=100` |
| **Payload schema** | `neo4j_recipe_id: str`, `recipe_name: str`, `chunk_index: int`, `source_page: int`, `text: str` |

### `recipe_images` Collection

| Field | Value |
|-------|-------|
| **Vector size** | 512 (CLIP ViT-B/32) |
| **Distance** | Cosine |
| **HNSW config** | `m=16`, `ef_construct=100` |
| **Payload schema** | `neo4j_recipe_id: str`, `recipe_name: str`, `image_path: str`, `source_page: int` |

---

## Retrieval Flow: Hybrid Query Processing

```
User Question
    │
    ▼
┌───────────────┐
│  LLM Router   │   Classify → GRAPH_ONLY / VECTOR_ONLY / HYBRID
└───────┬───────┘
        │
        ▼ (if GRAPH or HYBRID)
┌───────────────┐
│ Graph Retriever│   NL → Cypher → Neo4j → Recipe IDs
└───────┬───────┘
        │ filtered IDs
        ▼
┌───────────────┐
│Vector Retriever│   Semantic search in Qdrant (filtered by IDs)
└───────┬───────┘    Returns: text chunks + image refs
        │
        ▼
┌───────────────┐
│ LLM Synthesizer│   Graph context + Vector context → Answer with [citations]
└───────┬───────┘
        │
        ▼
┌──────────────────────────────────────┐
│ JSON Response                        │
│ {                                    │
│   "response": "To make... [1]",     │
│   "citations": {                     │
│     "1": {"text": "...", "image": ""}│
│   }                                  │
│ }                                    │
└──────────────────────────────────────┘
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | React 18 + Vite | Chat UI with citation popups |
| API | FastAPI | REST endpoints, CORS, static files |
| Orchestration | LangChain | LLM chaining, prompt management |
| LLM | GPT-4o-mini | Entity extraction, routing, synthesis |
| Graph DB | Neo4j 5.x | Knowledge graph for logical queries |
| Vector DB | Qdrant 1.10 | Semantic search (text + images) |
| Text Embedding | BAAI/bge-m3 | 1024-dim multilingual embeddings |
| Image Embedding | CLIP ViT-B/32 | 512-dim image-text embeddings |
| PDF Processing | PyMuPDF | Text + image extraction |
| Containerization | Docker Compose | Neo4j + Qdrant services |
