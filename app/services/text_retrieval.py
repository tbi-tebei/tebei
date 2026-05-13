import csv
import random
import os
from app.core.config import settings


class TextRetriever:
    def __init__(self):
        self.image_ids: list[str] = []
        self._load()

    def _load(self):
        path = settings.CAPTIONS_FILE
        if not os.path.exists(path):
            return
        seen = set()
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                if row:
                    img = row[0].strip()
                    if img and img not in seen:
                        seen.add(img)
                        self.image_ids.append(img)

    def search(self, query: str, top_k: int = 12) -> list[str]:
        # MOCK — replace with real retrieval later
        if not self.image_ids:
            return []
        k = min(top_k, len(self.image_ids))
        return random.sample(self.image_ids, k)
