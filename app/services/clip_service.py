import io
import json
import os

import faiss
import numpy as np
from PIL import Image

from app.core.config import settings

MODEL_NAME = "clip-ViT-B-32"


class CLIPService:
    """Singleton wrapper around a CLIP model and a FAISS vector index.

    A single ``clip-ViT-B-32`` model encodes both text and images into the
    same 512-dimensional space, which halves memory usage compared to running
    separate text and image models.

    The index uses ``IndexFlatIP`` (exact inner-product search). Because all
    vectors are L2-normalised before insertion, inner product equals cosine
    similarity, giving exact nearest-neighbour results with 100 % recall.
    """

    def __init__(self):
        self._model = None
        self._index: faiss.Index | None = None
        self._matrix: np.ndarray | None = None
        self._image_ids: list[str] = []

    def _load(self):
        """Lazy-initialise the CLIP model and load the FAISS index from disk.

        Subsequent calls are no-ops once the model is in memory.
        """
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(MODEL_NAME, device="cpu")
        self._load_index()

    def _load_index(self):
        """Read the FAISS index, image-ID list, and raw vector matrix from disk.

        The raw matrix (``image_matrix.npy``) is optional; when absent, Rocchio
        negative feedback is silently disabled.
        """
        index_path = os.path.join(settings.INDEX_DIR, "image_index.faiss")
        ids_path = os.path.join(settings.INDEX_DIR, "image_ids.json")
        matrix_path = os.path.join(settings.INDEX_DIR, "image_matrix.npy")
        if os.path.exists(index_path) and os.path.exists(ids_path):
            self._index = faiss.read_index(index_path)
            with open(ids_path, encoding="utf-8") as f:
                self._image_ids = json.load(f)
        if os.path.exists(matrix_path):
            self._matrix = np.load(matrix_path)

    def _write_index(self):
        """Persist the current FAISS index and image-ID list to disk."""
        os.makedirs(settings.INDEX_DIR, exist_ok=True)
        index_path = os.path.join(settings.INDEX_DIR, "image_index.faiss")
        ids_path = os.path.join(settings.INDEX_DIR, "image_ids.json")
        if self._index is None:
            return
        faiss.write_index(self._index, index_path)
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self._image_ids, f, ensure_ascii=False)

    def _new_index(self, dim: int) -> faiss.IndexFlatIP:
        """Create a new empty exact inner-product index for ``dim``-dim vectors."""
        return faiss.IndexFlatIP(dim)

    def reload_index(self):
        """Drop the in-memory index and reload it from disk."""
        self._index = None
        self._matrix = None
        self._image_ids = []
        self._load_index()

    def is_ready(self) -> bool:
        """Return True when the model is loaded and the index has at least one vector."""
        self._load()
        return self._index is not None and self._index.ntotal > 0

    def _search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        """Run exact inner-product search and return (image_id, cosine_score) pairs.

        Vectors stored in the index are L2-normalised, so inner product equals
        cosine similarity directly.
        """
        if not self.is_ready():
            return []
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_emb.astype(np.float32), k)
        return [
            (self._image_ids[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]

    def get_vectors(self, indices: list[int]) -> np.ndarray:
        """Reconstruct stored vectors from the FAISS index by position.

        Used by the Rocchio algorithm to compute positive and negative centroids.
        """
        return np.vstack([self._index.reconstruct(int(i)) for i in indices if i >= 0])

    def add_image(self, image_bytes: bytes, image_id: str):
        """Encode an image and insert it into the live FAISS index.

        The updated index is persisted to disk immediately so uploads survive
        a container restart.
        """
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
        """Encode a text query and return the top-k most similar images."""
        self._load()
        emb = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return self._search(emb, top_k)

    def search_negative(self, query_emb: np.ndarray, top_k: int) -> list[int]:
        """Return the indices of the ``top_k`` most dissimilar vectors.

        Uses brute-force dot product over the raw matrix to find the vectors
        furthest from the query. These become the negative examples in Rocchio.
        Falls back to an empty list when ``image_matrix.npy`` is not available.
        """
        if self._matrix is None:
            return []
        sims = (self._matrix @ query_emb.reshape(-1)).astype(np.float32)
        neg_indices = np.argpartition(sims, top_k)[:top_k]
        return [int(i) for i in neg_indices]

    def encode_text(self, query: str) -> np.ndarray:
        """Encode a single text string into a normalised CLIP embedding."""
        self._load()
        return self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

    def encode_texts(self, queries: list[str]) -> np.ndarray:
        """Encode a batch of text strings into normalised CLIP embeddings."""
        self._load()
        return self._model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)

    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        """Encode raw image bytes into a normalised CLIP embedding."""
        self._load()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._model.encode([img], convert_to_numpy=True, normalize_embeddings=True)

    def search_by_embedding(self, emb: np.ndarray, top_k: int = 12) -> list[tuple[str, float]]:
        """Search the FAISS index with a precomputed embedding vector."""
        return self._search(emb, top_k)


clip_service = CLIPService()
