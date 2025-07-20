# Literature Review — Domain-Specific Multimodal RAG

> Personal research notes compiled during the design phase of this project.
> These notes shaped the fundamental architectural decisions.

---

## 1. RAG Foundations

**Paper**: *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* — Lewis et al. (2020)

### Core Insight

Large Language Models are powerful reasoning machines but suffer from two fatal flaws
when deployed on domain-specific tasks:

| Problem | Description |
|---------|-------------|
| **Hallucination** | LLMs confidently fabricate facts when their parametric knowledge is insufficient |
| **Stale Knowledge** | Training data has a cutoff date; new information is invisible to the model |

RAG solves both by decoupling **memory** (retrieval corpus) from **reasoning** (generator).
The retriever fetches relevant passages from an external knowledge base, and the generator
conditions its output on those passages.

### Key Takeaway for This Project

For a cooking/nutrition domain, the LLM has no reliable built-in knowledge about our
specific recipe corpus. Without RAG, it would hallucinate recipe steps, invent ingredient
quantities, or confuse similar dishes. RAG grounds every answer in our actual PDF sources.

### Architecture Pattern (from the paper)

```
Query → Retriever(query, corpus) → Top-K documents → Generator(query, docs) → Answer
```

The retriever uses dense embeddings (DPR in the paper, BGE-M3 in our project) to find
semantically similar passages. The generator (GPT-4o-mini in our case) consumes the
retrieved context to produce a faithful answer.

---

## 2. Constraint-Based Knowledge Graph Filtering — Why Vector Search Alone Fails

**Reference**: Neo4j Graph Filtering techniques (Note: This is strictly Constraint-based Filtering, NOT Microsoft's GraphRAG which requires hierarchical Leiden clustering).

### The Blind Spots of Pure Vector RAG

Vector similarity search excels at finding passages that are semantically *close* to a
query. However, it catastrophically fails on:

| Failure Mode | Example | Why Vector Fails |
|-------------|---------|------------------|
| **Negation** | "Recipes without pork" | Embedding of "without pork" is close to "pork" |
| **Multi-hop reasoning** | "Japanese dishes that are both spicy and vegan" | Requires composing multiple constraints |
| **Aggregation** | "How many recipes use tofu?" | Vector search retrieves *similar* docs, not *all* |
| **Structural queries** | "What ingredients do ramen and udon share?" | Requires graph traversal, not similarity |

### Knowledge Graph as the Solution

A Knowledge Graph (KG) represents data as **entities** (nodes) and **relationships** (edges).
This structure naturally supports:

- **Exact filtering**: `(:Recipe)-[:HAS_TAG]->(:Tag {name: "Vegan"})` — zero ambiguity
- **Negation**: `WHERE NOT (r)-[:CONTAINS_INGREDIENT]->(:Ingredient {name: "Pork"})` — trivial in Cypher
- **Multi-hop**: Traverse multiple relationship types in a single query
- **Aggregation**: `COUNT`, `COLLECT`, path operations — native graph operations

### Hybrid Architecture Decision

Our system uses **both**:

1. **Neo4j** handles hard logical constraints (tags, ingredients, cuisine type, exclusions)
2. **Qdrant** handles soft semantic similarity (recipe instructions, similar cooking techniques)
3. **Graph-first filtering** ensures the vector search space is pre-filtered and hallucination-free

```
Query → Router → [Graph Filter] → Filtered IDs → [Vector Search(filtered)] → Context → LLM
```

---

## 3. HNSW Algorithm — How Qdrant Searches at Scale

**Reference**: Qdrant documentation, *"Efficient and robust approximate nearest neighbor search using HNSW"* — Malkov & Yashunin (2018)

### The Problem

Given a query vector, find the K most similar vectors in a collection of millions.
Brute-force (comparing against every vector) is O(n) — far too slow for production.

### HNSW: Hierarchical Navigable Small World

Qdrant uses HNSW as its primary indexing algorithm. The key ideas:

1. **Multi-layer graph**: Vectors are organized into a hierarchy of proximity graphs.
   Top layers are sparse (long-range connections), bottom layers are dense (local connections).

2. **Greedy search**: Start at the top layer, greedily move to the nearest neighbor,
   then descend to finer layers. Each layer refinement narrows the search space.

3. **Logarithmic complexity**: Search time is O(log n) instead of O(n).

### Key Parameters (tuned in our project)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `m` | 16 | Max connections per node in the graph. Higher = better recall, more memory |
| `ef_construct` | 100 | Build-time search depth. Higher = better index quality, slower build |
| `ef` (query-time) | 128 | Search-time depth. Higher = better recall, slower query |

### RAM Consumption Math (Capacity Planning)
For production scale, Qdrant's HNSW vector index memory footprint can be calculated as:
`Memory = N_vectors * (dim * 4 + M * 4) * 1.5`
For 50 million text vectors (dim=1024, M=16):
`RAM = 50,000,000 * (1024 * 4 + 16 * 4) * 1.5 = 50M * (4096 + 64) * 1.5 = ~312 GB`

### Why This Matters for Our Project

Our recipe corpus may contain thousands of text chunks and images. HNSW ensures that
even at scale, vector retrieval completes in single-digit milliseconds — essential for
a responsive chat interface.

### Practical Consideration

HNSW provides **approximate** nearest neighbors. The `ef` parameter controls the
accuracy/speed tradeoff. For our use case (cooking recipes, not life-critical), the
default approximation quality is more than sufficient.

---

## 4. LangChain & Qdrant Integration Notes

### LangChain

- **Role**: Orchestration framework. Chains together LLM calls, retrieval, and output parsing.
- **Key abstractions used**:
  - `ChatOpenAI` — wrapper around GPT-4o-mini
  - `StructuredOutputParser` — for entity extraction JSON
  - `GraphCypherQAChain` — translates natural language to Cypher (Neo4j)
  - Custom `Runnable` chains for the hybrid retrieval pipeline

### Qdrant

- **Role**: Vector database storing text and image embeddings
- **Collections**:
  - `recipe_text` — BGE-M3 embeddings (dim=1024), stores recipe instruction chunks
  - `recipe_images` — CLIP embeddings (dim=512), stores recipe photo vectors
- **Payload filtering**: Each vector carries a `neo4j_recipe_id` payload field,
  enabling filtered search scoped to graph-retrieved recipe IDs
- **Distance metric**: Cosine similarity (normalized vectors)

### BGE-M3 Embedding Model

- Multilingual, multi-granularity model from BAAI
- 1024-dimensional dense vectors
- Strong performance on both English and multilingual retrieval benchmarks
- Selected for potential Vietnamese/Japanese recipe text handling

### CLIP for Image Embeddings

- OpenAI CLIP maps images and text into a shared embedding space
- Enables cross-modal retrieval: text query → similar recipe images
- Using ViT-B/32 variant for balance of quality and inference speed
