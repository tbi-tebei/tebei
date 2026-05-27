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


HNSW_EF_SEARCH = 64


class CLIPService:
    def __init__(self):
        self._model = None
        self._index: faiss.Index | None = None
        self._matrix: np.ndarray | None = None
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
        matrix_path = os.path.join(settings.INDEX_DIR, "image_matrix.npy")
        if os.path.exists(index_path) and os.path.exists(ids_path):
            self._index = faiss.read_index(index_path)
            self._index.hnsw.efSearch = HNSW_EF_SEARCH
            with open(ids_path) as f:
                self._image_ids = json.load(f)
        if os.path.exists(matrix_path):
            self._matrix = np.load(matrix_path)

    def reload_index(self):
        self._index = None
        self._matrix = None
        self._image_ids = []
        self._load_index()

    def is_ready(self) -> bool:
        self._load()
        return self._index is not None and self._index.ntotal > 0

    def _search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Search HNSW index and convert L2 distances to cosine similarity."""
        if not self.is_ready():
            return []
        k = min(top_k, self._index.ntotal)
        l2_dists, indices = self._index.search(query_emb.astype(np.float32), k)
        # For L2-normalized vectors: cosine_sim = 1 - (l2_dist_sq / 2)
        return [
            (self._image_ids[idx], float(1.0 - dist / 2.0))
            for dist, idx in zip(l2_dists[0], indices[0])
            if idx >= 0
        ]

    def get_vectors(self, indices: list[int]) -> np.ndarray:
        """Reconstruct stored vectors from the FAISS index by position."""
        return np.vstack([self._index.reconstruct(int(i)) for i in indices if i >= 0])

    def search_negative(self, query_emb: np.ndarray, top_k: int) -> list[int]:
        """Find the most dissimilar vectors using brute-force dot product."""
        if self._matrix is None:
            return []
        sims = (self._matrix @ query_emb.reshape(-1)).astype(np.float32)
        neg_indices = np.argpartition(sims, top_k)[:top_k]
        return [int(i) for i in neg_indices]

    def encode_text(self, query: str) -> np.ndarray:
        """Encode a text string into a CLIP embedding."""
        self._load()
        return self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """Encode raw image bytes into a CLIP embedding."""
        self._load()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._model.encode([img], convert_to_numpy=True, normalize_embeddings=True)

    def search_by_embedding(self, emb: np.ndarray, top_k: int = 12) -> list[tuple[str, float]]:
        """Search the FAISS index with a precomputed embedding vector."""
        return self._search(emb, top_k)


clip_service = CLIPService()
