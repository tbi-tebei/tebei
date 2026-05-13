import csv
import random
import os
from app.core.config import settings


class TextRetriever:
    def __init__(self):
        self.image_ids: list[str] = []
        self.captions: dict[str, list[str]] = {}
        self._load()

    def _load(self):
        path = settings.CAPTIONS_FILE
        if not os.path.exists(path):
            return
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
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

    def search(self, query: str, top_k: int = 12) -> list[dict]:
        # MOCK — replace with real retrieval later
        if not self.image_ids:
            return []
        k = min(top_k, len(self.image_ids))
        sampled = random.sample(self.image_ids, k)

        # generate descending mock scores
        scores = sorted([random.uniform(0.5, 1.0) for _ in sampled], reverse=True)

        return [
            {
                "image_id": img,
                "score": round(score, 4),
                "caption": self.captions[img][0],
            }
            for img, score in zip(sampled, scores)
        ]
