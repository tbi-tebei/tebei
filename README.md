# tebei — Text-Image Search Engine

A photo search engine that lets you find images using natural language queries or similar images. Built for the Information Retrieval course project (2025–2026).

**Live demo:** https://tebei.hafizmuh.site/

---

## Features

| Feature | Status |
|---|---|
| Text → Image search | **Active** — CLIP `clip-ViT-B-32` + Rocchio query expansion + cross-encoder reranker |
| Image → Image search | **Active** — CLIP image encoder + Rocchio query expansion |
| Multilingual query | **Active** — automatic translation via Google Translate |
| Dataset | Flickr30k — 31,784 images |

---

## How it works

1. **Indexing** — all images are encoded into 512-dim vectors using CLIP (`clip-ViT-B-32`) and stored in a FAISS `IndexFlatIP` index (cosine similarity).
2. **Query expansion** — at search time, the query vector is refined using pseudo-relevance feedback via the Rocchio algorithm (two-pass retrieval).
3. **Reranking** (text search only) — top candidates are reranked using a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) that scores each (query, caption) pair for relevance.

### Retrieval pipeline

```
Text query
    ↓
CLIP text encoder → 512-dim query vector
    ↓
Rocchio refinement (pseudo-relevance feedback)
    ↓
FAISS k-NN search → top candidates
    ↓
Cross-encoder reranker (query × caption)
    ↓
Final results
```

---

## Stack

- **Backend**: FastAPI + Uvicorn
- **Embedding model**: `sentence-transformers` (CLIP ViT-B/32)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Vector index**: FAISS (`IndexFlatIP`)
- **Runtime**: Docker (Python 3.11)

### Scalability note

The current deployment uses FAISS (in-process vector search) and local filesystem storage. This is sufficient for single-instance usage but does not support horizontal scaling. For a production-grade deployment, FAISS can be replaced with OpenSearch or Elasticsearch (both support k-NN vector search natively) to enable concurrent multi-user access and persistent distributed indexing.

---

## Getting started

### Prerequisites
- Docker and Docker Compose
- Flickr30k dataset ([Kaggle](https://www.kaggle.com/datasets/srinivasac/flickr30k-dataset))

### Setup

```bash
# 1. Place dataset files
mkdir -p data/raw/Images
# Copy all images to data/raw/Images/
# Copy captions.txt to data/raw/captions.txt

# 2. Build FAISS index (run once)
# Option A: locally (slow, CPU)
docker-compose run --rm app python scripts/build_index.py

# Option B: Google Colab / Kaggle (recommended, GPU)
# Open notebooks/build_index_colab.ipynb and follow instructions
# Download output files to data/index/

# 3. Build and run
docker-compose build
docker-compose up
```

Open http://localhost:8001

---

## Project structure

```
app/
├── main.py
├── core/config.py
├── models/schemas.py
├── api/routes/
│   ├── search.py
│   └── upload.py
├── services/
│   ├── clip_service.py       # CLIP model + FAISS index (singleton)
│   ├── query_expansion.py    # Rocchio pseudo-relevance feedback
│   ├── reranker.py           # Cross-encoder reranker
│   ├── text_retrieval.py
│   ├── image_retrieval.py
│   └── data_store.py
└── templates/index.html
data/
├── raw/                      # Dataset (gitignored)
└── index/                    # FAISS index files (gitignored)
scripts/
└── build_index.py            # Offline: encode image → FAISS index
notebooks/
└── build_index_colab.ipynb   # Alternative: build index in Google Colab
```

---

## API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/api/search/text` | Text → image search |
| POST | `/api/search/image` | Image → image search |
| GET | `/api/health` | Health check |
| GET | `/docs` | Swagger UI (auto-generated) |

---

## Known limitations

- Cross-encoder reranker relies on captions for scoring. Non-English queries are automatically translated to English before reranking. If reranker confidence is low (score < 0.1), results fall back to CLIP scores.
- FAISS index is not thread-safe under concurrent writes; bulk uploads are not supported.
- Index must be rebuilt via `build_index.py` if the dataset changes significantly.

---

## Team

| Name | NPM |
|---|---|
| Muhammad Hafiz | 2206082732 |
| Ratu Nadya Anjania | 2206029752 |
| Wahyu Hidayat | 2206081894 |