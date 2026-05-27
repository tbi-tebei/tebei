import csv
import logging
import os
from collections import Counter

import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi

from app.core.config import settings

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

logger = logging.getLogger(__name__)


class PatriciaNode:
    """Node in a Patricia tree (compressed trie / radix tree)."""
    __slots__ = ("prefix", "children", "terms")

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.children: dict[str, PatriciaNode] = {}
        self.terms: list[str] = []


class PatriciaTrie:
    """Compressed trie where single-child chains are merged into one node."""

    def __init__(self):
        self._root = PatriciaNode()

    def insert(self, term: str):
        key = term.lower()
        node = self._root
        i = 0
        while i < len(key):
            ch = key[i]
            if ch not in node.children:
                leaf = PatriciaNode(key[i:])
                leaf.terms.append(term)
                node.children[ch] = leaf
                return
            child = node.children[ch]
            p = child.prefix
            j = 0
            while j < len(p) and i < len(key) and p[j] == key[i]:
                j += 1
                i += 1
            if j < len(p):
                # Split: shared prefix up to j, then diverge
                split = PatriciaNode(p[:j])
                split.terms = child.terms[:]
                child.prefix = p[j:]
                split.children[p[j]] = child
                if i < len(key):
                    new_leaf = PatriciaNode(key[i:])
                    new_leaf.terms.append(term)
                    split.children[key[i]] = new_leaf
                split.terms.append(term)
                node.children[ch] = split
                return
            child.terms.append(term)
            node = child
        node.terms.append(term)

    def _collect(self, node: PatriciaNode, limit: int) -> list[str]:
        results = list(node.terms[:limit])
        if len(results) >= limit:
            return results[:limit]
        for child in node.children.values():
            results.extend(self._collect(child, limit - len(results)))
            if len(results) >= limit:
                break
        return results[:limit]

    def search(self, prefix: str, limit: int = 8) -> list[str]:
        key = prefix.lower()
        node = self._root
        i = 0
        while i < len(key):
            ch = key[i]
            if ch not in node.children:
                return []
            child = node.children[ch]
            p = child.prefix
            j = 0
            while j < len(p) and i < len(key) and p[j] == key[i]:
                j += 1
                i += 1
            if i < len(key) and j >= len(p):
                node = child
                continue
            if i >= len(key):
                return child.terms[:limit]
            return []
        return node.terms[:limit]


class DataStore:
    """In-memory store for image IDs, captions, and BM25 index."""

    def __init__(self):
        self.image_ids: list[str] = []
        self.captions: dict[str, list[str]] = {}
        self.bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self.suggestions: list[str] = []
        self._trie = PatriciaTrie()
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

        self._trie = PatriciaTrie()
        for term, _ in word_freq.most_common(500):
            self._trie.insert(term)
        for term, _ in bigram_freq.most_common(200):
            self._trie.insert(term)
        logger.info("Built %d suggestions, trie loaded", len(self.suggestions))

    def autocomplete(self, prefix: str, limit: int = 8) -> list[str]:
        """Return terms matching the given prefix via Patricia tree lookup."""
        prefix = prefix.lower().strip()
        if not prefix:
            return []
        return self._trie.search(prefix, limit)

    def add_image(self, image_id: str, caption: str):
        if image_id not in self.captions:
            self.captions[image_id] = []
            self.image_ids.append(image_id)
        self.captions[image_id].append(caption)
        with open(settings.CAPTIONS_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([image_id, caption])


data_store = DataStore()
