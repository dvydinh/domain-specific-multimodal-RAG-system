# Literature Review - Domain-Specific Multimodal RAG

> Research notes compiled during the design phase. Every citation below
> links to a verifiable academic paper or official documentation.

---

## 1. Retrieval-Augmented Generation (RAG) — Foundations

**Paper:** Lewis, P., et al. (2020). *"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."* NeurIPS 2020.
**Link:** [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

### Core Insight

LLMs suffer from hallucination (fabricating facts) and knowledge staleness (training cutoff). RAG decouples **memory** (external retrieval corpus) from **reasoning** (generator), grounding every answer in retrieved evidence.

| Problem | Description |
|---------|-------------|
| **Hallucination** | LLMs confidently generate plausible but incorrect information |
| **Stale Knowledge** | Parametric knowledge cannot be updated without retraining |

### Application to This Project

Our recipe corpus is proprietary PDF content. Without RAG, a general LLM would hallucinate ingredient quantities or confuse similar dishes. RAG grounds every response in actual PDF-extracted text.

**Architecture pattern:**
```
Query → Dense Retriever(query, corpus) → Top-K passages → Generator(query, docs) → Cited Answer
```

---

## 2. Dense Passage Retrieval & BGE-M3

**Paper:** Karpukhin, V., et al. (2020). *"Dense Passage Retrieval for Open-Domain Question Answering."* EMNLP 2020.
**Link:** [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)

**Paper:** Chen, J., et al. (2024). *"BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation."* ACL 2024.
**Link:** [arXiv:2402.03216](https://arxiv.org/abs/2402.03216)

### Why BGE-M3 for This Project

- 1024-dimensional dense vectors with state-of-the-art retrieval quality
- Multi-lingual capability (critical for potential Vietnamese/Japanese recipe text)
- Multi-granularity: supports both short queries and long passage encoding
- Outperforms E5, GTE, and older SBERT models on MTEB benchmarks

---

## 3. Knowledge Graphs for Constraint-Based Filtering

**Paper:** Ji, S., et al. (2022). *"A Survey on Knowledge Graphs: Representation, Acquisition, and Applications."* IEEE TNNLS, 33(2), 494–514.
**Link:** [arXiv:2002.00388](https://arxiv.org/abs/2002.00388)

**Note on GraphRAG:** Microsoft's GraphRAG (Edge, D., et al., 2024, [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)) focuses on hierarchical community summarization via Leiden clustering. Our approach is fundamentally different — we use Neo4j as a **constraint-based pre-filter**, not for summarization.

### Why Vector Search Alone Fails

| Failure Mode | Example | Root Cause |
|:---|:---|:---|
| Negation | "Recipes without pork" | Embedding("without pork") ≈ Embedding("pork") |
| Multi-constraint | "Japanese, spicy, and vegan" | Requires composing 3 filters simultaneously |
| Aggregation | "How many recipes use tofu?" | Vector retrieves *similar* docs, not *all* docs |
| Structural | "Ingredients shared by ramen and udon?" | Requires graph traversal, not similarity |

### Our Hybrid Architecture

```
Query → Router → [Neo4j Constraint Filter] → Filtered Recipe IDs → [Qdrant Vector Search] → LLM
```

Neo4j handles exact ingredient/tag constraints; Qdrant ranks within the constrained scope.

---

## 4. CLIP and Cross-Modal Retrieval

**Paper:** Radford, A., et al. (2021). *"Learning Transferable Visual Models From Natural Language Supervision."* ICML 2021.
**Link:** [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

### Dual-Encoder Architecture

CLIP trains a text encoder and an image encoder jointly, mapping both modalities into a shared 512-dimensional vector space. This enables:

- **Text → Image:** "spicy ramen" query retrieves photos of ramen dishes
- **Image → Image:** Similar recipe photos cluster together
- **Cross-modal grounding:** Recipe instructions + recipe photos linked via the same vector space

### Implementation in This Project

- Text chunks: BGE-M3 (dim=1024) → `recipe_text` collection
- Recipe images: OpenCLIP ViT-B/32 (dim=512) → `recipe_images` collection
- Both collections carry `neo4j_recipe_id` payload for graph-vector cross-referencing

---

## 5. HNSW: Approximate Nearest Neighbor Search

**Paper:** Malkov, Y. & Yashunin, D. (2018). *"Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs."* IEEE TPAMI, 42(4), 824–836.
**Link:** [arXiv:1603.09320](https://arxiv.org/abs/1603.09320)

### Algorithm Summary

HNSW constructs a multi-layer navigable small-world graph:
1. Top layers: sparse, long-range connections (coarse navigation)
2. Bottom layers: dense, local connections (fine-grained search)
3. Greedy traversal from top to bottom achieves O(log N) search complexity

### Qdrant Configuration (this project)

| Parameter | Value | Effect |
|:---|:---|:---|
| `m` | 16 | Max connections per node |
| `ef_construct` | 100 | Build-time search depth |

### Memory Footprint Formula
```
RAM = N_vectors × (dim × 4 + m × 4) × 1.5
```
For 50M text vectors (dim=1024, m=16): ~312 GB

---

## 6. Outbox Pattern for Distributed Consistency

**Reference:** Richardson, C. (2018). *Microservices Patterns.* Manning Publications. Ch. 4: "Managing transactions with sagas."

**Reference:** Kleppmann, M. (2017). *Designing Data-Intensive Applications.* O'Reilly. Ch. 9: "Consistency and Consensus."

### Application to This Project

Our system writes to two independent stores (Neo4j + Qdrant). Without coordination, a crash mid-write creates phantom data. The **Transactional Outbox pattern** (Saga) ensures:

1. Write intent is persisted to SQLite BEFORE any mutation
2. If Phase 2 (Qdrant) fails, a background worker detects the stuck transaction and alerts for cleanup
3. The SQLite outbox survives process crashes (unlike in-memory dicts)

---

## 7. Server-Sent Events for Streaming LLM Responses

**Specification:** W3C. (2015). *"Server-Sent Events."* [W3C Recommendation](https://www.w3.org/TR/eventsource/).

### Rationale

RAG pipelines typically take 3-8 seconds end-to-end. Synchronous REST responses create a "dead loading" UX. SSE enables:
- Immediate metadata delivery (retrieval stats)
- Token-by-token streaming of the LLM response
- Reduced perceived latency from seconds to milliseconds
