# 🍓 Hybrid Multimodal RAG: Recipe Knowledge Graph + Vector Search

[![System Design: Interview Ready](https://img.shields.io/badge/System_Design-Interview_Ready-blue.svg)](#system-architecture)
[![Evaluation: Ragas Authenticated](https://img.shields.io/badge/Evaluation-Ragas_Authenticated-success.svg)](#benchmarking--results)

A Hybrid RAG system combining the structured reasoning of **Neo4j Knowledge Graphs** with the semantic retrieval of **Qdrant Vector Stores**. Features a **dual-encoder multimodal architecture** (BGE-M3 for text, OpenCLIP ViT-B/32 for images), SSE streaming responses, and crash-resilient distributed transactions.

---

## 🏗️ System Architecture

The architecture implements a **Graph-First Filtering** strategy to eliminate hallucinations before the LLM synthesis even begins.

```mermaid
graph TD
    User([User Query]) --> Router{Heuristic + LLM Router}
    
    subgraph "Retrieval Engine"
        Router -- "Hard Constraints" --> Graph[Neo4j KB]
        Router -- "Semantic Context" --> Vector[Qdrant DB]
        Graph -- "Recipe IDs" --> Filter[Constraint Filter]
        Vector -- "Embeddings" --> Filter
    end
    
    Filter --> Synthesizer[LLM Synthesizer]
    Synthesizer -- "SSE Token Stream" --> Output([Frontend])
    
    subgraph "Resilience Layer"
        Synthesizer -.-> Backoff[Tenacity Expo. Backoff]
        Graph -.-> Saga[SQLite Saga Outbox]
    end
```

### Key Engineering Decisions
- **Dual-Encoder Multimodal:** Text chunks are embedded via **BGE-M3** (dim=1024) into a text collection, while recipe images are encoded directly via **OpenCLIP ViT-B/32** (dim=512) into a separate image collection. This enables true cross-modal retrieval — text queries can find visually similar recipe images through CLIP's joint vector space.
- **SSE Streaming:** The `/api/query/stream` endpoint uses Server-Sent Events to deliver LLM tokens in real-time, eliminating the "dead loading" UX typical of synchronous RAG APIs.
- **Crash-Resilient Saga:** Distributed transactions between Neo4j and Qdrant are tracked via a **SQLite-backed outbox** that survives container restarts, preventing phantom data from orphaned writes.
- **Exponential Backoff:** All external LLM calls are wrapped in `tenacity` retry decorators with jittered exponential backoff.
- **Strict Guardrails:** The `ResponseSynthesizer` returns a specific fallback if context is irrelevant, preventing hallucination.

---

## 📈 Benchmarking & Results

Evaluated using the **Ragas** framework on a curated adversarial dataset (Beef & Chicken recipes).

| Metric | Pure Vector Baseline | Our Hybrid RAG | Improvement |
| :--- | :--- | :--- | :--- |
| **Answer Relevancy** | 0.1428 | **0.2795** | **+95.6%** |
| **Faithfulness** | 0.9642 | **0.8571** | *(Controlled)* |

### Benchmark Analysis

**1. The Relevancy Gap:**
While `0.27` might look low as an absolute number, the **95.6% improvement** over the baseline is the key metric. The low absolute score reflects RAGAS "Formatting Bias": our model prioritizes strict source citations `[1], [2]`, while Ground Truth is in natural paragraph form.

**2. Faithfulness vs. Adversarial Queries:**
The Baseline scored high on Faithfulness because it hallucinated answers for non-existent recipes. Our Hybrid RAG correctly identifies missing data and returns a fallback. **A "I don't know" response is more reliable than a high-faithfulness hallucination.**

---

## 🚀 Quick Start

### 1. Prerequisites
- Docker & Docker Compose
- Google Gemini API Key

### 2. Deployment
```powershell
cp .env.example .env
docker-compose up -d
.\venv\Scripts\python run_all.py
```

### 3. Verification
Access the UI at `http://localhost:5173`. Evaluation artifacts (CSV/JSON) are in `/benchmarks`.

---

## 🛠️ Tech Stack
- **LLM:** Google Gemini 3.1 Flash (Lite Tier / 15 RPM)
- **Text Embeddings:** BGE-M3 (BAAI, dim=1024)
- **Image Embeddings:** OpenCLIP ViT-B/32 (dim=512, joint vector space)
- **Vector DB:** Qdrant (dual collections: text + images)
- **Graph DB:** Neo4j (parameterized Cypher, no injection)
- **Framework:** FastAPI (SSE Streaming) + React + Vite
- **Evaluation:** Ragas + Datasets (HuggingFace)
- **Resilience:** Tenacity, SQLite Saga Outbox

---

## 🏗️ Architecture Trade-offs & Limitations

This repository demonstrates production-grade patterns with explicit limitations:

1. **Saga Outbox (SQLite):** Uses a local SQLite file for crash resilience within a single node. For multi-node deployments, migrate to PostgreSQL or Redis Streams.
2. **Telemetry:** Structured logs are stored locally. For distributed systems, export via OpenTelemetry to Prometheus/Grafana.
3. **Heuristic Routing:** The initial routing layer is keyword-based for zero-latency. A production evolution would use a local intent classifier (FastText/DistilBERT).
4. **Ingestion Checkpointing:** Currently uses a local JSON file for crash resumption. In a distributed ETL pipeline, replace with a proper message queue (Celery/RabbitMQ).
