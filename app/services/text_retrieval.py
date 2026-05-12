from typing import List
from app.models.schemas import SearchResult


class TextRetriever:
    """
    Plug-in text retrieval. Replace the body of `search` with your chosen method:
      - BM25  : pip install rank-bm25
      - TF-IDF: sklearn.feature_extraction.text.TfidfVectorizer
      - Dense : sentence-transformers + FAISS
    """

    def __init__(self):
        self.index = None
        self._load_index()

    def _load_index(self):
        # TODO: deserialize and load your text index from disk
        # Example BM25:
        #   import pickle
        #   with open("data/index/bm25.pkl", "rb") as f:
        #       self.index = pickle.load(f)
        pass

    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        if self.index is None:
            return []

        # TODO: run your retrieval method and return SearchResult objects
        # Example BM25:
        #   tokenized = query.lower().split()
        #   scores = self.index.get_scores(tokenized)
        #   top_idx = scores.argsort()[-top_k:][::-1]
        #   return [
        #       SearchResult(id=str(i), score=float(scores[i]), text=self.corpus[i])
        #       for i in top_idx
        #   ]
        return []
