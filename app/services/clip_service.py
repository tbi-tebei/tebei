import io
import json
import os

import faiss
import numpy as np
from PIL import Image

from app.core.config import settings

# Single model for both text and image — saves ~600MB RAM vs two-model approach.
# clip-ViT-B-32 maps both text (English) and images to the same 512-dim space.
MODEL_NAME = "clip-ViT-B-32"


class CLIPService:
    def __init__(self):
        self._model = None
        self._index: faiss.Index | None = None
        self._image_ids: list[str] = []

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(MODEL_NAME, device="cpu")
        self._load_index()

    def _load_index(self):
        index_path = os.path.join(settings.INDEX_DIR, "image_index.faiss")
        ids_path = os.path.join(settings.INDEX_DIR, "image_ids.json")
        if os.path.exists(index_path) and os.path.exists(ids_path):
            self._index = faiss.read_index(index_path)
            with open(ids_path, encoding="utf-8") as f:
                self._image_ids = json.load(f)

    def _write_index(self):
        os.makedirs(settings.INDEX_DIR, exist_ok=True)
        index_path = os.path.join(settings.INDEX_DIR, "image_index.faiss")
        ids_path = os.path.join(settings.INDEX_DIR, "image_ids.json")
        if self._index is None:
            return
        faiss.write_index(self._index, index_path)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self._image_ids, f, ensure_ascii=False)

    def _new_index(self, dim: int) -> faiss.IndexFlatIP:
        return faiss.IndexFlatIP(dim)

    def reload_index(self):
        self._index = None
        self._image_ids = []
        self._load_index()

    def is_ready(self) -> bool:
        self._load()
        return self._index is not None and self._index.ntotal > 0

    def _search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if not self.is_ready():
            return []
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_emb.astype(np.float32), k)
        return [
            (self._image_ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def add_image(self, image_bytes: bytes, image_id: str):
        self._load()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        emb = self._model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)
        if self._index is None:
            self._index = self._new_index(emb.shape[1])
        self._index.add(emb)
        self._image_ids.append(image_id)
        self._write_index()

    def search_by_text(self, query: str, top_k: int = 12) -> list[tuple[str, float]]:
        self._load()
        emb = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return self._search(emb, top_k)

    def search_by_image(self, image_bytes: bytes, top_k: int = 12) -> list[tuple[str, float]]:
        self._load()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        emb = self._model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
        return self._search(emb, top_k)


clip_service = CLIPService()
