import numpy as np
from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """Cross-encoder reranker that scores (query, caption) text pairs.

    Uses ``ms-marco-MiniLM-L-6-v2``, a lightweight cross-encoder fine-tuned
    for passage re-ranking on MS MARCO. Raw logits are converted to
    probabilities via a sigmoid so scores fall in the (0, 1) range.
    """

    def __init__(self):
        self._model = None

    def _load(self):
        """Lazy-load the cross-encoder model on first use."""
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
        """Re-score and reorder candidates using cross-encoder inference.

        Each candidate is paired with its primary caption as a
        (query, caption) input to the cross-encoder. The model scores all
        pairs in a single batch, applies a sigmoid, and returns the top-k
        results sorted by descending score.
        """
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
