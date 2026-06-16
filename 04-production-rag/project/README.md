# Core Engineering Project — Hybrid GraphRAG Pipeline

## Objective
Build a production-grade hybrid RAG pipeline combining graph traversal, dense vector
search, advanced chunking, and query enhancement, gated by RAGAS evaluation.

## Deliverables
1. **Entity–relationship graph** (Neo4j) from a document corpus; community detection
   (Leiden); generated community summaries.
2. **Fusion pipeline**: graph-traversal context + HNSW vector search merged via RRF;
   parent-child chunking + HyDE query enhancement.
3. **RAGAS evaluation**: context relevance, faithfulness, answer relevance — set as
   CI/CD pipeline gate with minimum thresholds.
4. **Benchmark**: single-hop vs multi-hop queries — GraphRAG vs naive RAG vs hybrid
   search on identical question sets.

## Acceptance Checklist
- [ ] Entity graph built and community detection produces coherent summaries
- [ ] RRF fusion correctly combines graph context + vector search results
- [ ] Parent-child chunking demonstrably reduces hallucination vs flat chunking
- [ ] HyDE measurably improves recall on domain-specific sparse queries
- [ ] RAGAS triad implemented and wired as a CI/CD gate (with thresholds documented)
- [ ] Multi-hop benchmark shows GraphRAG advantage where naive RAG fails
- [ ] Single-hop benchmark confirms no regression vs naive RAG

## Results
Place final report, plots, and tables in `results/`.
