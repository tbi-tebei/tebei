import csv
import logging
import os
from collections import Counter

from rank_bm25 import BM25Okapi

from app.core.config import settings

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "in", "on", "at", "to",
    "of", "and", "or", "for", "with", "by", "from", "up", "as", "it", "its",
    "that", "this", "has", "have", "had", "be", "been", "being", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "can", "shall",
    "not", "no", "but", "if", "so", "than", "too", "very", "just", "about",
    "into", "over", "after", "before", "between", "through", "during", "while",
    "out", "off", "down", "then", "there", "here", "where", "when", "what",
    "who", "which", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "also", "back",
    "even", "still", "already", "since", "he", "she", "they", "him", "her",
    "his", "their", "them", "we", "our", "you", "your", "i", "me", "my",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "many", "much", "another", "next", "new", "old", "big", "small",
    "large", "long", "little", "well", "way", "because", "like", "look",
    "looking", "looks", "appears", "wearing", "near", "front", "behind",
    "around", "along", "across", "against", "above", "below", "beside",
    "under", "inside", "outside", "left", "right", "side", "top",
}

logger = logging.getLogger(__name__)


class DataStore:
    """In-memory store for image IDs, captions, and BM25 index."""

    def __init__(self):
        self.image_ids: list[str] = []
        self.captions: dict[str, list[str]] = {}
        self.bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self.suggestions: list[str] = []
        self._word_list: list[str] = []
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
        self._build_suggestions()

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

    def _build_suggestions(self):
        """Extract popular terms from captions for search suggestions."""
        word_freq = Counter()
        bigram_freq = Counter()
        for img_id in self.image_ids:
            for caption in self.captions.get(img_id, []):
                words = [w.strip(".,!?;:'\"()") for w in caption.lower().split()]
                words = [w for w in words if w and w not in STOP_WORDS and len(w) > 2]
                word_freq.update(words)
                for a, b in zip(words, words[1:]):
                    bigram_freq[f"{a} {b}"] += 1

        top_bigrams = [term for term, _ in bigram_freq.most_common(20)]
        top_words = [term for term, _ in word_freq.most_common(30)
                     if not any(term in bg for bg in top_bigrams)]
        self.suggestions = top_bigrams[:10] + top_words[:10]

        self._word_list = [term for term, _ in word_freq.most_common(500)]
        self._word_list.extend(term for term, _ in bigram_freq.most_common(200))
        logger.info("Built %d suggestions, %d autocomplete terms", len(self.suggestions), len(self._word_list))

    def autocomplete(self, prefix: str, limit: int = 8) -> list[str]:
        """Return terms matching the given prefix."""
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        return [t for t in self._word_list if t.startswith(prefix)][:limit]

    def add_image(self, image_id: str, caption: str):
        if image_id not in self.captions:
            self.captions[image_id] = []
            self.image_ids.append(image_id)
        self.captions[image_id].append(caption)
        with open(settings.CAPTIONS_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([image_id, caption])


data_store = DataStore()
