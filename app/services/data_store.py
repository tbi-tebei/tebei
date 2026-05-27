import csv
import logging
import os

from rank_bm25 import BM25Okapi

from app.core.config import settings

logger = logging.getLogger(__name__)


class DataStore:
    """In-memory store for image IDs, captions, and BM25 index."""

    def __init__(self):
        self.image_ids: list[str] = []
        self.captions: dict[str, list[str]] = {}
        self.bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self.load()

    def load(self):
        self.image_ids.clear()
        self.captions.clear()
        path = settings.CAPTIONS_FILE
        if not os.path.exists(path):
            return
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                img = row[0].strip()
                caption = row[1].strip() if len(row) > 1 else ""
                if not img:
                    continue
                if img not in self.captions:
                    self.captions[img] = []
                    self.image_ids.append(img)
                self.captions[img].append(caption)
        self._build_bm25()

    def _build_bm25(self):
        """Build BM25 index from all captions for lexical search."""
        docs = []
        self._bm25_ids = []
        for img_id in self.image_ids:
            text = " ".join(self.captions.get(img_id, []))
            docs.append(text.lower().split())
            self._bm25_ids.append(img_id)
        if docs:
            self.bm25 = BM25Okapi(docs)
            logger.info("BM25 index built: %d documents", len(docs))

    def search_bm25(self, query: str, top_k: int = 12) -> list[tuple[str, float]]:
        """Keyword search over captions using BM25."""
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(query.lower().split())
        top_indices = scores.argsort()[::-1][:top_k]
        return [
            (self._bm25_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    def add_image(self, image_id: str, caption: str):
        if image_id not in self.captions:
            self.captions[image_id] = []
            self.image_ids.append(image_id)
        self.captions[image_id].append(caption)
        with open(settings.CAPTIONS_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([image_id, caption])


data_store = DataStore()
