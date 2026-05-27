# tebei — Text-Image Search Engine

A personal photo search engine that lets you find images using natural language queries or similar images. Built for the Information Retrieval course project (2025-2026).

**Live demo:** https://tebei.hafizmuh.site/

---

## Features

| Feature | Description |
|---|---|
| Text-to-image search | Hybrid retrieval: semantic (CLIP + FAISS HNSW) + lexical (BM25) with RRF fusion |
| Image-to-image search | Find visually similar images using CLIP embeddings with pseudo-relevance feedback |
| LLM query expansion | Gemini generates visual descriptions to improve abstract queries |
| Pseudo-relevance feedback | Rocchio algorithm with positive and negative feedback refines query vectors |
| Cross-encoder reranking | Reranks candidates by scoring (query, caption) pairs for semantic relevance |
| Multilingual queries | Automatic translation to English via Google Translate |
| Hybrid search | Combines semantic vector search and BM25 lexical search using Reciprocal Rank Fusion |
| Search suggestions | Auto-generated from caption frequency, with autocomplete via Patricia tree |
| Image upload | Add personal photos to the index with captions |
| Rate limiting | Per-IP request throttling to ensure fair concurrent access |

---

## Retrieval pipeline

### Text-to-image search

![Text Search Pipeline](docs/text-search-pipeline.svg)

### Image-to-image search

![Image Search Pipeline](docs/image-search-pipeline.svg)

---

## Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Embedding model | CLIP ViT-B/32 (sentence-transformers) |
| Vector index | FAISS IndexHNSWFlat (M=32, efConstruction=128, efSearch=64) |
| Lexical search | BM25 Okapi (rank-bm25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| LLM expansion | Gemini 2.5 Flash |
| Translation | deep-translator (Google Translate) |
| Autocomplete | Patricia tree (compressed trie) |
| Rate limiting | slowapi |
| Stopwords | NLTK English stopwords |
| Runtime | Docker (Python 3.11) |
| Dataset | Flickr30k (31,783 images with captions) |

---

## Scalability

| Aspect | Implementation |
|---|---|
| Vector search | FAISS HNSW (O(log n) approximate nearest neighbor) |
| Lexical search | BM25 in-process (rank-bm25) |
| Autocomplete | Patricia tree (compressed trie, O(k) prefix lookup) |
| Concurrent access | Per-IP rate limiting (slowapi) |
| Model loading | Preloaded at startup, cached |
| LLM calls | Response caching with lru_cache(256) |
| Containerization | Docker + docker-compose |

---

## Getting started

### Prerequisites

- Docker and Docker Compose
- Flickr30k dataset ([Kaggle](https://www.kaggle.com/datasets/srinivasac/flickr30k-dataset))
- Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### Setup

```bash
# 1. Place dataset files
mkdir -p data/raw/Images
# Copy all images to data/raw/Images/
# Copy captions.txt to data/raw/captions.txt

# 2. Create .env file
echo "GEMINI_API_KEY=your-key-here" > .env

# 3. Build FAISS index (run once, ~20 min on CPU)
docker-compose run --rm app python scripts/build_index.py

# 4. Build and run
docker-compose build
docker-compose up
```

Open http://localhost:8001

---

## Project structure

```
app/
├── main.py                       # FastAPI app, model preloading, rate limiting
├── core/config.py                # Settings (env vars, paths)
├── models/schemas.py             # Pydantic request/response models
├── api/routes/
│   ├── search.py                 # Text search, image search, suggestions, autocomplete
│   └── upload.py                 # Image upload endpoint
├── services/
│   ├── clip_service.py           # CLIP model + FAISS HNSW index + negative search
│   ├── query_expansion.py        # PRF (Rocchio) + LLM expansion (Gemini)
│   ├── reranker.py               # Cross-encoder reranker
│   ├── text_retrieval.py         # Full text search pipeline (hybrid + rerank)
│   ├── image_retrieval.py        # Image search pipeline (PRF)
│   └── data_store.py             # Captions, BM25 index, Patricia tree, suggestions
├── static/css/style.css
├── static/js/app.js
└── templates/index.html
data/
├── raw/                          # Dataset (gitignored)
└── index/
    ├── image_index.faiss         # HNSW vector index
    ├── image_matrix.npy          # Raw vectors for negative feedback
    └── image_ids.json            # Image ID mapping
scripts/
└── build_index.py                # Encodes images with CLIP, builds HNSW index
```

---

## API

| Method | Path | Description | Rate limit |
|---|---|---|---|
| GET | `/` | Web UI | - |
| POST | `/api/search/text` | Text-to-image search | 15/min per IP |
| POST | `/api/search/image` | Image-to-image search | 10/min per IP |
| GET | `/api/search/suggestions` | Popular search terms | 30/min per IP |
| GET | `/api/search/autocomplete?q=` | Prefix autocomplete | 30/min per IP |
| POST | `/api/upload/image` | Upload image with caption | 30/min per IP |
| GET | `/api/health` | Health check | - |
| GET | `/docs` | Swagger UI (auto-generated) | - |

---

## Known limitations

- Cross-encoder reranker scores (query, caption) text pairs. Results with missing or short captions may rank lower regardless of visual relevance.
- LLM query expansion depends on Gemini API availability. If the API is unreachable, the system falls back to PRF-only retrieval.
- FAISS HNSW provides approximate nearest neighbors (~99% recall). For exact results, switch to IndexFlatIP in `build_index.py`.
- BM25 lexical search uses simple whitespace tokenization. A production system would use proper tokenization and stemming.
- Index must be rebuilt via `build_index.py` after switching index types or changing the dataset.

---

## Team

| Name | NPM |
|---|---|
| Muhammad Hafiz | 2206082732 |
| Ratu Nadya Anjania | 2206029752 |
| Wahyu Hidayat | 2206081894 |
