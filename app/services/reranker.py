from sentence_transformers import CrossEncoder
import numpy as np

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        self._model = CrossEncoder(MODEL_NAME)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        captions: dict[str, list[str]],
        top_k: int = 12,
    ) -> list[tuple[str, float]]:
        self._load()
        if not candidates:
            return []

        pairs = [
            (query, captions.get(img_id, [""])[0])
            for img_id, _ in candidates
        ]
        scores = self._model.predict(pairs)
        scores = 1 / (1 + np.exp(-scores))
        reranked = sorted(
            zip([img_id for img_id, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(img_id, float(score)) for img_id, score in reranked[:top_k]]

reranker = Reranker()