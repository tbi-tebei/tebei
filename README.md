# Text-Image Search Engine
Information Retrieval Final Project

## Setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000

## Project Structure

```
app/
├── main.py                  # FastAPI app + frontend routes
├── core/config.py           # Settings (loaded from .env)
├── models/schemas.py        # Pydantic request/response models
├── api/routes/
│   ├── search.py            # POST /api/search/text  &  /api/search/image
│   └── index.py             # POST /api/index/build  &  GET /api/index/status
├── services/
│   ├── text_retrieval.py    # BM25 / TF-IDF / dense text retrieval
│   ├── image_retrieval.py   # CLIP / multimodal retrieval
│   └── indexer.py           # Builds and persists the index
├── templates/index.html     # Jinja2 frontend
└── static/                  # CSS + JS
data/
├── raw/                     # Your dataset goes here (gitignored)
└── index/                   # Built index files (gitignored)
notebooks/                   # Exploration / experiments
tests/
└── test_search.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| POST | `/api/search/text` | Text search (`{"query": "...", "top_k": 10}`) |
| POST | `/api/search/image` | Image/multimodal search (form-data: `query`, `image`, `top_k`) |
| POST | `/api/index/build` | Build the search index |
| GET | `/api/index/status` | Index status |
| GET | `/docs` | Swagger UI |

## Implementation Guide

1. **Choose your retrieval approach** — edit `app/services/text_retrieval.py` and/or `app/services/image_retrieval.py`
2. **Put your dataset** in `data/raw/`
3. **Implement `Indexer.build()`** in `app/services/indexer.py` to process the dataset
4. **Call `POST /api/index/build`** once to build the index
5. **Search!**

### Recommended options

| Goal | Library |
|------|---------|
| Classic text IR | `rank-bm25` (BM25) or scikit-learn (TF-IDF) |
| Text embeddings | `sentence-transformers` |
| Image embeddings | `open-clip-torch` (CLIP) |
| Fast ANN search | `faiss-cpu` |

Uncomment the relevant lines in `requirements.txt` and `pip install -r requirements.txt`.

## Tests

```bash
pytest tests/ -v
```
