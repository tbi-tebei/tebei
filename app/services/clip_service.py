import io
import json
import os

import faiss
import numpy as np
from PIL import Image

from app.core.config import settings

# clip-ViT-B-32-multilingual-v1 encodes multilingual text into CLIP's embedding space
# clip-ViT-B-32 encodes images — both produce 512-dim vectors in the same space
TEXT_MODEL_NAME = "clip-ViT-B-32-multilingual-v1"
IMAGE_MODEL_NAME = "clip-ViT-B-32"


class CLIPService:
    def __init__(self):
        self._text_model = None
        self._image_model = None
        self._index: faiss.Index | None = None
        self._image_ids: list[str] = []

    def _load(self):
        if self._text_model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._text_model = SentenceTransformer(TEXT_MODEL_NAME, device="cpu")
        self._image_model = SentenceTransformer(IMAGE_MODEL_NAME, device="cpu")
        self._load_index()

    def _load_index(self):
        index_path = os.path.join(settings.INDEX_DIR, "image_index.faiss")
        ids_path = os.path.join(settings.INDEX_DIR, "image_ids.json")
        if os.path.exists(index_path) and os.path.exists(ids_path):
            self._index = faiss.read_index(index_path)
            with open(ids_path) as f:
                self._image_ids = json.load(f)

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

    def search_by_text(self, query: str, top_k: int = 12) -> list[tuple[str, float]]:
        self._load()
        emb = self._text_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return self._search(emb, top_k)

    def search_by_image(self, image_bytes: bytes, top_k: int = 12) -> list[tuple[str, float]]:
        self._load()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        emb = self._image_model.encode([img], convert_to_numpy=True, normalize_embeddings=True)
        return self._search(emb, top_k)


clip_service = CLIPService()
