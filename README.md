# domain-specific multimodal RAG system

A hybrid retrieval-augmented generation (RAG) system combining a Neo4j knowledge graph with a Qdrant vector database. The system enforces strict constraint filtering before semantic search, mitigating hallucinations often seen in pure-vector architectures.

## architecture

```
user query
   │
   ▼
[ llm router ] ── classify ──► graph / vector / hybrid
   │
   ▼ (if constraints detected)
[ neo4j graph ] ── filter ──► exact logical constraints (ingredients, tags, exclusions)
   │
   ▼ (filtered IDs)
[ qdrant vector ] ── search ──► semantic matching on text & image (scoped by graph)
   │
   ▼
[ llm synthesis ] ── respond ──► context-aware formulation with exact citations
```

### design rationale
- **graph-first filtering:** instead of relying on embedding distance for absolute exclusions ("without pork"), the router translates hard constraints to cypher queries. The vector store only searches within the pre-approved scope.
- **streaming ingestion:** the text extraction and embedding pipelines run on python generators (`yield`), chunking and indexing pages continuously to prevent OOM errors on large PDF corpora.
- **on-disk payloads:** Qdrant is configured with `on_disk_payload=True`, shifting the raw string storage overhead from RAM completely to disk arrays (RocksDB). RAM is reserved solely for HNSW vector matrices.
- **read-only retrieval:** LLM-generated cypher queries are sandboxed. The execution engine enforces `session.execute_read()`, physically blocking runtime cypher injection (e.g. `DETACH DELETE`).

## tech stack

- **graph database:** Neo4j 5.x
- **vector database:** Qdrant 1.10
- **text embeddings:** BAAI/bge-m3 (1024-dim)
- **image embeddings:** CLIP ViT-B/32 (512-dim)
- **llm interface:** OpenAI GPT-4o-mini
- **backend:** FastAPI, Uvicorn, LangChain
- **frontend:** React 18, Vite
- **testing:** Pytest, Ragas

## deployment

The application is containerized using multi-stage Docker builds.

1. Ensure the `.env` file is populated using `.env.example`.
2. Start the production stack:

```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

This will initialize:
- `neo4j` (port 7474/7687)
- `qdrant` (port 6333)
- `api` (the backend server)
- `web` (nginx serving compiled frontend, accessible on port 80)

## local development

To run the application natively without dockerized code syncing overhead:

1. Bring up the databases:
   ```bash
   docker-compose up -d
   ```
2. Setup and run the backend:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   uvicorn backend.api.main:app --reload
   ```
3. Run the frontend (Vite HMR):
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

## ingestion pipeline

To ingest new PDF cookbooks into the databases:

```bash
python -m backend.ingestion.pipeline data/raw/
```

## dataset evaluation

Ragas scripts test hallucination ratios exclusively against a dedicated hold-out set (`data/sample/eval_recipes.json`) to prevent data leakage from training distribution.

```bash
python tests/evaluate_ragas.py
```

## license
MIT
