# tebei — Text-Image Search Engine

> A multimodal search engine that retrieves images using natural language queries or similar images, powered by CLIP embeddings, FAISS vector indexing, BM25 lexical search, LLM query expansion, and cross-encoder reranking.

**Live demo:** https://tebei.hafizmuh.site/

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-purple)
![FAISS](https://img.shields.io/badge/FAISS-IndexFlatIP-orange)

---

## Team

| Name | NPM |
|---|---|
| Muhammad Hafiz | 2206082732 |
| Ratu Nadya Anjania | 2206029752 |
| Wahyu Hidayat | 2206081894 |

---


## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Development](#development)
- [Design Decisions](#design-decisions)
- [Known Limitations](#known-limitations)
- [Team](#team)

---

## Overview

**tebei** is a full-stack multimodal information retrieval system built as the final project for the Information Retrieval course (2025–2026). It allows users to:

- Search a corpus of ~31,000 Flickr images using free-form natural language
- Find visually similar images by uploading a query image
- Expand the index by uploading new images with captions

The system combines multiple retrieval strategies — semantic vector search, BM25 lexical search, LLM query expansion, Rocchio pseudo-relevance feedback, and cross-encoder reranking — into a single hybrid pipeline.

---

## Features

| Feature | Description |
|---|---|
| **Text-to-image search** | Hybrid retrieval combining CLIP semantic search and BM25 lexical search, fused with Reciprocal Rank Fusion (RRF) |
| **Image-to-image search** | Visual similarity search using CLIP image embeddings with Rocchio pseudo-relevance feedback |
| **LLM query expansion** | Gemini 2.5 Flash generates 3 visual descriptions per query; CLIP encodes them and averages vectors to enrich the query representation |
| **Pseudo-relevance feedback** | Rocchio algorithm refines the query vector using the top-K retrieved results as positive examples and the most dissimilar vectors as negative examples |
| **Cross-encoder reranking** | A `ms-marco-MiniLM-L-6-v2` cross-encoder rescores `(query, caption)` pairs and reorders final results |
| **Multilingual queries** | Automatic language detection and translation to English via Google Translate before encoding |
| **Autocomplete** | Patricia tree (compressed trie) over all caption tokens, providing O(k) prefix lookup |
| **Search suggestions** | Curated popular terms derived from caption token frequency at index build time |
| **Image upload** | Users can add new images with captions; the system encodes and inserts them into the live FAISS index |
| **Rate limiting** | Per-IP throttling via slowapi to ensure fair access under concurrent load |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        Client (Browser)                  │
│           Vanilla JS + Fetch API + Jinja2 template       │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
┌────────────────────────▼────────────────────────────────┐
│                    FastAPI Application                    │
│                                                          │
│  ┌──────────────┐   ┌──────────────┐  ┌──────────────┐  │
│  │ /api/search  │   │ /api/upload  │  │  Rate Limiter│  │
│  └──────┬───────┘   └──────┬───────┘  └──────────────┘  │
│         │                  │                              │
│  ┌──────▼───────────────────▼────────────────────────┐   │
│  │                    Services Layer                  │   │
│  │                                                    │   │
│  │  TextRetriever      ImageRetriever   DataStore     │   │
│  │  ├─ Translation     ├─ CLIP encode   ├─ Captions   │   │
│  │  ├─ LLM expansion   └─ Rocchio PRF   ├─ BM25 index │   │
│  │  ├─ Rocchio PRF                      └─ Patricia   │   │
│  │  ├─ RRF fusion                           tree      │   │
│  │  └─ Cross-encoder                                  │   │
│  │      reranking                                     │   │
│  └────────────────────────────────────────────────────┘   │
│                                                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │               CLIPService (shared)                  │  │
│  │   SentenceTransformer(clip-ViT-B-32)                │  │
│  │   FAISS IndexFlatIP  │  Raw vector matrix (NumPy)   │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │      data/index/    │
              │  image_index.faiss  │
              │  image_ids.json     │
              │  image_matrix.npy   │
              └─────────────────────┘
```

---

## Retrieval Pipeline

### Text-to-Image Search

```
User query
    │
    ▼
[1] Auto-translate to English (Google Translate)
    │
    ▼
[2] LLM Query Expansion (Gemini 2.5 Flash)
    │  Generates 3 visual descriptions
    │  Encodes query + descriptions → averages CLIP vectors
    │
    ▼
[3] Pseudo-Relevance Feedback (Rocchio)
    │  positive  = top-10 most similar vectors (FAISS search)
    │  negative  = top-10 most dissimilar vectors (brute-force dot product)
    │  refined   = α·q + β·mean(positive) - γ·mean(negative)
    │  α=1.0, β=0.75, γ=0.15
    │
    ▼──────────────────────────────────────┐
    │  Semantic search                     │  Lexical search
    │  CLIP embedding → FAISS IndexFlatIP  │  BM25 Okapi (rank-bm25)
    │  cosine similarity (L2-normalized)   │  term-frequency scoring
    └──────────────┬───────────────────────┘
                   │
                   ▼
           [4] RRF Fusion  (k=60)
               score(d) = Σ 1/(k + rank_i(d))
                   │
                   ▼
           [5] Cross-Encoder Reranking
               ms-marco-MiniLM-L-6-v2
               scores (query, caption) pairs
               falls back to RRF order if top score < 0.1
                   │
                   ▼
               Top-K results
```

![Text Search Pipeline](docs/text-search-pipeline.svg)

### Image-to-Image Search

```
Query image (bytes)
    │
    ▼
[1] CLIP image encoder → 512-dim embedding
    │
    ▼
[2] Pseudo-Relevance Feedback (Rocchio)
    │  Same formula as text pipeline
    │
    ▼
[3] FAISS IndexFlatIP cosine search
    │
    ▼
Top-K visually similar images
```

![Image Search Pipeline](docs/image-search-pipeline.svg)

---

## Tech Stack

| Component | Technology | Reason |
|---|---|---|
| **Backend framework** | FastAPI + Uvicorn | Async, fast, auto-generates OpenAPI docs |
| **Embedding model** | CLIP ViT-B/32 (sentence-transformers) | Single model encodes both text and images into the same 512-dim space |
| **Vector index** | FAISS IndexFlatIP | Exact inner-product search on L2-normalized vectors = exact cosine similarity |
| **Lexical search** | BM25 Okapi (rank-bm25) | Complements semantic search; handles rare keywords and exact matches |
| **Reranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-attention over `(query, caption)` pairs provides more accurate relevance signal than bi-encoder scores |
| **LLM expansion** | Gemini 2.5 Flash | Generates visual descriptions to enrich abstract queries |
| **Translation** | deep-translator (Google Translate) | Enables multilingual queries without a separate multilingual CLIP |
| **Autocomplete** | Patricia tree (compressed trie) | O(k) prefix lookup where k is the query length; memory-efficient over all caption tokens |
| **Rate limiting** | slowapi | Per-IP throttling with configurable limits per endpoint |
| **Stopwords** | NLTK English stopwords | Filters noise words from BM25 and autocomplete |
| **Configuration** | pydantic-settings | Type-safe settings from environment variables |
| **Containerization** | Docker + docker-compose | Reproducible deployment |
| **Dataset** | Flickr30k (31,783 images, 5 captions each) | Standard IR benchmark with diverse real-world photography |

---

## Getting Started

### Prerequisites

- Python 3.10+ (for local dev) or Docker
- Flickr30k dataset ([Kaggle](https://www.kaggle.com/datasets/srinivasac/flickr30k-dataset))
- Gemini API key from [Google AI Studio](https://aistudio.google.com/) *(optional — falls back to PRF-only if absent)*

### Local Development

```bash
# 1. Clone and create virtual environment
git clone <repo-url>
cd TK
python3 -m venv env
source env/bin/activate        # Windows: env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset
mkdir -p data/raw/Images
# Copy all .jpg images to data/raw/Images/
# Copy captions.txt to data/raw/captions.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your-key-here

# 5. Build the FAISS index (runs once, ~20 min on CPU)
python scripts/build_index.py

# 6. Start the development server (hot-reload)
make run
# → http://localhost:8000
```

### Docker (Production)

```bash
# 1. Place dataset files (same as step 3 above)

# 2. Create .env with your Gemini API key
cp .env.example .env

# 3. Build the index inside Docker
docker-compose run --rm app python scripts/build_index.py

# 4. Build and start the service
docker-compose build
docker-compose up -d

# → http://localhost:8001
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `Text-Image Search Engine` | Application name |
| `DATA_DIR` | `data/raw` | Root directory for raw dataset |
| `INDEX_DIR` | `data/index` | Directory for FAISS index and metadata |
| `TOP_K` | `10` | Default number of results returned |
| `MAX_UPLOAD_SIZE` | `5242880` | Maximum upload size in bytes (5 MB) |
| `GEMINI_API_KEY` | *(empty)* | Gemini API key for LLM query expansion |

---

## Project Structure

```
tebei/
├── app/
│   ├── main.py                  # FastAPI app entry point; model preloading; rate limiter
│   ├── core/
│   │   └── config.py            # Pydantic settings (env vars → typed Python)
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── api/routes/
│   │   ├── search.py            # GET/POST endpoints: text search, image search,
│   │   │                        #   suggestions, autocomplete
│   │   └── upload.py            # POST /api/upload/image
│   └── services/
│       ├── clip_service.py      # Singleton: CLIP model + FAISS index + encode/search
│       ├── data_store.py        # Captions map, BM25 index, Patricia tree, suggestions
│       ├── query_expansion.py   # Rocchio PRF + LLM expansion (Gemini); LRU-cached
│       ├── text_retrieval.py    # Full text pipeline: translate → expand → fuse → rerank
│       ├── image_retrieval.py   # Image pipeline: encode → PRF → search
│       └── reranker.py          # Cross-encoder (ms-marco-MiniLM-L-6-v2)
├── app/static/
│   ├── css/style.css            # Dark-theme UI styles
│   └── js/app.js                # Vanilla JS: search, upload, autocomplete, modal
├── app/templates/
│   └── index.html               # Jinja2 template (single-page UI)
├── app/public/
│   └── tebei-logo.png
├── scripts/
│   └── build_index.py           # Encodes all Flickr30k images with CLIP;
│                                #   writes FAISS index + image_ids.json + image_matrix.npy
├── docs/
│   ├── text-search-pipeline.svg
│   └── image-search-pipeline.svg
├── tests/
│   ├── conftest.py              # Pytest fixtures (TestClient, mock services)
│   └── test_search.py           # Integration tests for search and upload endpoints
├── data/
│   ├── raw/                     # Dataset files (gitignored)
│   │   ├── Images/              # ~31k .jpg files
│   │   └── captions.txt         # CSV: image_name, comment (5 rows per image)
│   └── index/                   # Built by build_index.py (gitignored)
│       ├── image_index.faiss    # FAISS IndexFlatIP (512-dim, ~31k vectors)
│       ├── image_ids.json       # Ordered list of image filenames
│       └── image_matrix.npy     # Raw float32 matrix for Rocchio negative feedback
├── .env.example                 # Environment variable template
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
└── pyproject.toml               # Ruff linter config
```

---

## API Reference

All endpoints return JSON. The base URL is `http://localhost:8000`.

### `POST /api/search/text`

Search images using a natural language query.

**Request body** (`application/json`):
```json
{
  "query": "a dog running on the beach",
  "top_k": 12
}
```

**Response**:
```json
{
  "query": "a dog running on the beach",
  "total": 12,
  "results": [
    {
      "image_id": "3637013_4d4b3c7c2a.jpg",
      "score": 0.4821,
      "caption": "A brown dog runs across a sandy beach.",
      "image_url": "/images/3637013_4d4b3c7c2a.jpg"
    }
  ]
}
```

**Rate limit:** 15 requests / minute per IP

---

### `POST /api/search/image`

Search for visually similar images by uploading a query image.

**Request body** (`multipart/form-data`):
| Field | Type | Description |
|---|---|---|
| `image` | file | Query image (JPG, PNG, WEBP, JPE, JFIF) |
| `top_k` | int | Number of results (default: 12) |

**Response:** same schema as `/api/search/text`

**Rate limit:** 10 requests / minute per IP

---

### `GET /api/search/suggestions`

Returns a curated list of popular search terms derived from caption frequency.

**Response**:
```json
{ "suggestions": ["dog", "woman", "playing", "field", "bicycle"] }
```

---

### `GET /api/search/autocomplete?q={prefix}`

Returns autocomplete completions for the given prefix using the Patricia tree.

**Query params:** `q` — prefix string (minimum 2 characters)

**Response**:
```json
{ "results": ["running", "runner", "runs"] }
```

---

### `POST /api/upload/image`

Upload a new image with a caption. The image is encoded with CLIP and inserted into the live FAISS index immediately.

**Request body** (`multipart/form-data`):
| Field | Type | Description |
|---|---|---|
| `image` | file | Image to upload (JPG, PNG, WEBP, JPE, JFIF; max 5 MB) |
| `caption` | string | Text description of the image |

**Response**:
```json
{
  "image_id": "a1b2c3d4...jpg",
  "image_url": "/images/a1b2c3d4...jpg",
  "caption": "A cat sitting on a windowsill."
}
```

**Rate limit:** 30 requests / minute per IP

---

### `GET /api/health`

Health check endpoint for uptime monitoring.

**Response:** `{ "status": "ok" }`

---

### `GET /docs`

Auto-generated Swagger UI (provided by FastAPI). Available at `/docs`.

---

## Development

### Running Tests

```bash
make test
# or
pytest tests/ -v
```

### Linting

The project uses [ruff](https://docs.astral.sh/ruff/) for fast Python linting.

```bash
make lint
# or
ruff check app/ tests/
```

### Makefile Commands

```bash
make install      # Create venv and install dependencies
make run          # Start dev server with hot-reload on :8000
make test         # Run pytest
make lint         # Run ruff linter
make build-index  # Trigger index rebuild via API (server must be running)
```

---

## Design Decisions

**Single CLIP model for text and images.** CLIP's key property is that it maps both modalities into the same embedding space. Using one `clip-ViT-B-32` model for all encoding halves GPU/CPU memory usage compared to running separate models (~600 MB vs ~1.2 GB).

**IndexFlatIP over HNSW.** FAISS `IndexFlatIP` performs exact inner-product search. Because all embeddings are L2-normalized before insertion, inner product equals cosine similarity. For a corpus of ~31k vectors (a manageable size), exact search is fast enough that approximate methods like HNSW provide no meaningful latency benefit, while exact search guarantees 100% recall.

**Negative feedback in Rocchio.** Standard Rocchio uses only positive pseudo-relevance feedback. Adding negative feedback (`GAMMA × mean(bottom-K vectors)`) pushes the refined query vector away from visually dissimilar images, which visibly improves retrieval precision on ambiguous queries.

**LRU cache on Gemini calls.** Since LLM expansion is the slowest step (~1–2 s), the same query string always produces the same prompt and near-identical output. Caching with `lru_cache(maxsize=256)` eliminates redundant API calls for repeated or popular queries.

**Cross-encoder fallback.** If the top cross-encoder score falls below 0.1 (sigmoid-transformed), it means all `(query, caption)` pairs are weak matches. In this case the system returns the RRF-fused ranking instead, which degrades more gracefully than a poor reranked list.

**Patricia tree for autocomplete.** A compressed trie stores only the branching nodes, reducing memory compared to a full trie. Prefix lookup is O(k) where k is the query length, independent of the corpus size — important when indexing hundreds of thousands of unique caption tokens.

---

## Known Limitations

- **Caption dependency.** The cross-encoder scores `(query, caption)` text pairs. Images with short or missing captions may rank lower regardless of visual relevance. Query expansion and CLIP semantic search partially compensate, but caption quality is a hard constraint.
- **LLM availability.** Query expansion depends on the Gemini API. If the API is unreachable or the key is missing, the system silently falls back to PRF-only CLIP search. Results are still reasonable but less precise for abstract queries.
- **English-only CLIP.** The underlying CLIP model was trained primarily on English text. Translation is applied before encoding, but translation errors can propagate to retrieval quality.
- **In-process BM25.** `rank-bm25` runs in memory and tokenizes on whitespace. A production system would benefit from proper stemming, an inverted index on disk, and support for phrase queries.
- **Index rebuild required.** Switching the FAISS index type or changing the dataset requires re-running `build_index.py`. The upload endpoint supports incremental insertion for new images only.


---

*Built for the Information Retrieval course, Faculty of Computer Science, Universitas Indonesia  2025/2026.*
