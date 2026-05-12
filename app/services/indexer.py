import os
from app.core.config import settings


class Indexer:
    """
    Builds and manages the search index over raw data.
    Implement `build` once you have decided on your retrieval approach.
    """

    def build(self) -> dict:
        # TODO: implement indexing pipeline
        # 1. Load documents from settings.DATA_DIR
        # 2. Extract text / images
        # 3. Compute embeddings or build BM25/TF-IDF index
        # 4. Save index to settings.INDEX_DIR
        os.makedirs(settings.INDEX_DIR, exist_ok=True)
        return {"docs_indexed": 0, "index_dir": settings.INDEX_DIR}

    def status(self) -> dict:
        index_ready = os.path.isdir(settings.INDEX_DIR) and bool(
            os.listdir(settings.INDEX_DIR)
        )
        return {
            "ready": index_ready,
            "docs_indexed": 0,
            "index_path": settings.INDEX_DIR,
        }
