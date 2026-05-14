# Text-Image Search Engine — TBI Final Project

Information Retrieval — search gambar menggunakan query teks atau gambar.

**Live:** https://tebei.hafizmuh.site/

---

## Status retrieval

| Fitur | Status |
|-------|--------|
| Text → Image search | **Aktif** — CLIP multilingual (`clip-ViT-B-32-multilingual-v1`) |
| Image → Image search | **Aktif** — CLIP image encoder (`clip-ViT-B-32`) |
| Multilingual query | **Aktif** — support Bahasa Indonesia & 50+ bahasa lain |
| Dataset | Flickr30k — 31.784 gambar |

Model text dan image di-embed ke ruang yang sama (512-dim). FAISS IndexFlatIP dipakai untuk cosine similarity search.

---

## Stack

- **Backend**: FastAPI + Uvicorn
- **Model**: sentence-transformers (CLIP multilingual)
- **Index**: FAISS
- **Runtime**: Docker (Python 3.11)

---

## Run (Docker)

```bash
# 1. Build image
docker-compose build

# 2. Build FAISS index dari dataset (sekali saja)
docker-compose run --rm app python scripts/build_index.py

# 3. Jalankan server
docker-compose up
```

Buka http://localhost:8001

---

## Project Structure

```
app/
├── main.py
├── core/config.py
├── models/schemas.py
├── api/routes/
│   ├── search.py            # POST /api/search/text, /api/search/image
│   └── upload.py
├── services/
│   ├── clip_service.py      # CLIP model + FAISS index (singleton)
│   ├── text_retrieval.py
│   ├── image_retrieval.py
│   └── data_store.py
└── templates/index.html
data/
├── raw/                     # Dataset (gitignored)
└── index/                   # FAISS index files (gitignored)
scripts/
└── build_index.py           # Offline: encode gambar → FAISS index
notebooks/
└── build_index_colab.ipynb  # Alternatif: build index di Google Colab (GPU)
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| POST | `/api/search/text` | Text → image search |
| POST | `/api/search/image` | Image → image search |
| GET | `/api/health` | Health check |
| GET | `/docs` | Swagger UI |
