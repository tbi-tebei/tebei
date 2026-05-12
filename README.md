# Text-Image Search Engine
Information Retrieval — Final Project

## Stack
- **Backend**: FastAPI
- **Frontend**: Jinja2 templates + vanilla JS
- **Python**: 3.11+

## Setup

```bash
python3 -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Run

```bash
uvicorn app.main:app --reload
# or
make run
```

Open http://localhost:8000

## Project Structure

```
app/
├── main.py                  # FastAPI entry point — add routes here
├── core/config.py           # App settings (loaded from .env)
├── models/schemas.py        # Pydantic request/response schemas
├── api/routes/
│   ├── search.py            # Search endpoints
│   └── index.py             # Index management endpoints
├── services/
│   ├── text_retrieval.py    # Text retrieval logic
│   ├── image_retrieval.py   # Image / multimodal retrieval logic
│   └── indexer.py           # Index builder
├── templates/index.html     # Frontend HTML
└── static/
    ├── css/style.css
    └── js/app.js
data/
├── raw/                     # Dataset (gitignored)
└── index/                   # Built index files (gitignored)
notebooks/                   # Experiments & exploration
tests/
└── test_search.py
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| GET | `/api/health` | Health check |
| GET | `/docs` | Swagger UI (auto-generated) |

## Common Commands

```bash
make run          # start dev server
make test         # run tests
make lint         # check code style
make build-index  # trigger index build (server must be running)
```

## Tests

```bash
pytest tests/ -v
```

## Dependencies

Uncomment the relevant lines in `requirements.txt` based on your retrieval approach, then re-run `pip install -r requirements.txt`.

| Approach | Library |
|----------|---------|
| BM25 | `rank-bm25` |
| TF-IDF | `scikit-learn` |
| Text/image embeddings | `sentence-transformers` |
| CLIP embeddings | `open-clip-torch` |
| Vector similarity search | `faiss-cpu` |
| Image I/O | `Pillow` |
