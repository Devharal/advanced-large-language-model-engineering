# Module 4 — Production-Grade Retrieval-Augmented Generation

> Engineer resilient, deterministic knowledge-retrieval systems combining hybrid
> search, semantic indexing, and graph topologies.

## Status
- [ ] Readiness map reviewed (`notes/00-readiness-map.md`)
- [ ] Part 1 notes + implementations
- [ ] Part 2 notes + implementations
- [ ] Part 3 notes + implementations
- [ ] Core Engineering Project complete

## Folder Structure
```
04-production-rag/
├── README.md
├── notes/
│   ├── 00-readiness-map.md
│   ├── 01-dense-sparse-hybrid-retrieval.md
│   ├── 02-vector-index-internals.md
│   ├── 03-advanced-chunking.md
│   ├── 04-query-enhancement-hyde.md
│   ├── 05-graphrag.md
│   └── 06-production-hardening-ragas.md
├── src/
│   ├── hybrid_retrieval.py     # FAISS HNSW + BM25Okapi + RRF
│   ├── vector_index_bench.py   # HNSW vs IVF
│   ├── chunking_strategies.py  # fixed/semantic/parent-child/agentic
│   ├── query_enhancement.py    # HyDE, self-querying, multi-query
│   ├── graphrag_pipeline.py    # Neo4j + community detection
│   └── ragas_eval.py
├── notebooks/
│   ├── recall_at_k_comparison.ipynb
│   ├── lost_in_the_middle.ipynb
│   └── graphrag_vs_naive_multihop.ipynb
├── data/
│   └── README.md               # corpus sourcing notes
└── project/
    ├── README.md
    └── results/
```

## Topics & Resource Directory

### Part 1 — Embedding & Search

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Dense & Sparse Retrieval | Bi-encoders, cross-encoder rerankers, BM25, RRF | FAISS HNSW + BM25Okapi index; RRF fusion; cross-encoder reranker pipeline |
| Vector Index Internals | HNSW, IVF, PCA dim reduction, vector drift | HNSW/IVF benchmarks at various `ef`; incremental re-indexing on drift |

**Resources:**
- Paper: *Dense Passage Retrieval for Open-Domain QA* (Karpukhin et al. 2020)
- Course: Stanford CS224N QA & Retrieval
- Repo: `facebookresearch/faiss`
- Blog: Pinecone *Complete Guide to Hybrid Search*; Pinecone HNSW explainer

### Part 2 — Chunking & Query Enhancement

| Topic | Key Concepts | What to Implement |
|---|---|---|
| Advanced Chunking | Fixed, semantic, parent-child, agentic chunking | All 4 strategies on same corpus; parent-child index in LlamaIndex; RAGAS faithfulness |
| Query Enhancement | Query rewriting, HyDE, self-querying, multi-query retrieval | HyDE pipeline; self-querying with metadata filters; 3-variation multi-query fusion |

**Resources:**
- Paper: *HyDE — Hypothetical Document Embeddings* (Gao et al. 2022)
- Course: LlamaIndex Advanced RAG workshop
- Repo: `run-llama/llama_index`
- Blog: LlamaIndex Advanced RAG Techniques series

### Part 3 — GraphRAG & Production Hardening

| Topic | Key Concepts | What to Implement |
|---|---|---|
| GraphRAG | Entity extraction, Leiden community detection, local vs global search | Neo4j entity graph + community detection; Microsoft GraphRAG; vs naive RAG on multi-hop |
| Production Hardening | "Lost in the middle", reranking, context pruning, RAGAS triad | Reproduce lost-in-the-middle + reranking fix; RAGAS CI/CD gate; async streaming RAG |

**Resources:**
- Paper: *From Local to Global: A GraphRAG Approach* (Edge et al., Microsoft 2024); *RAGAS* (Es et al. 2023)
- Course: CMU 11-711 RAG and Knowledge Graphs
- Repo: `microsoft/graphrag`; `explodinggradients/ragas`
- Blog: Microsoft Research GraphRAG blog; Jason Liu RAG evaluation newsletter

## Core Engineering Project
**Hybrid GraphRAG Pipeline** — see [`project/README.md`](project/README.md)
