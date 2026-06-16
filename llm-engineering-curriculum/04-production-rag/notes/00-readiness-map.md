# Section A — Module Readiness Map

## Part A1 — Prerequisite Dependency Tree

- **SEARCH & INFORMATION RETRIEVAL**
  - TF-IDF; BM25; inverted index
- **EMBEDDING FOUNDATIONS**
  - Word embeddings (Word2Vec, GloVe); sentence embeddings; cosine similarity
- **VECTOR DATABASE INTUITION**
  - ANN search; HNSW graph structure
- **BASIC RETRIEVAL PIPELINE**
  - Naive RAG: chunk → embed → store → retrieve → generate
  - Context window constraints

## YOU ARE NOW READY FOR
Dense/Sparse Hybrid → GraphRAG → Advanced Chunking → HyDE → RAGAS Evaluation

## Part A2 — Prerequisite Detail Table

| Concept | Why You Need It | Where to Learn It |
|---|---|---|
| BM25 keyword search | Hybrid RAG combines BM25 with dense via RRF | Robertson BM25 paper; Elastic BM25 explainer |
| Bi-encoder sentence embeddings | Foundation of all dense retrieval | SBERT paper; Sentence-Transformers docs |
| HNSW ANN | Vector index internals / recall-latency tradeoffs | Pinecone HNSW blog; FAISS docs |
| Naive RAG pipeline | Advanced RAG improves on this baseline | LlamaIndex RAG from scratch; LangChain RAG docs |
| Chunking & context windows | All chunking strategies solve this constraint | Anthropic context window docs; Greg Kamradt chunking tutorial |

## Checklist
- [ ] I can explain TF-IDF and BM25 scoring intuitively
- [ ] I understand bi-encoder vs cross-encoder tradeoffs
- [ ] I understand why ANN (HNSW) is needed over exact search at scale
- [ ] I can walk through the naive RAG pipeline end-to-end
- [ ] I understand why chunking exists (context window constraints)
