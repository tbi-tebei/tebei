from typing import List, Optional
from app.models.schemas import SearchResult


class ImageRetriever:
    """
    Plug-in multimodal retrieval. Replace the body of `search` with your chosen method:
      - CLIP  : pip install open-clip-torch  (or sentence-transformers[clip])
      - FAISS : pip install faiss-cpu
    """

    def __init__(self):
        self.model = None
        self.faiss_index = None
        self._load_model()

    def _load_model(self):
        # TODO: load your embedding model and FAISS index
        # Example CLIP via open_clip:
        #   import open_clip, faiss, torch
        #   self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        #   self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        #   self.faiss_index = faiss.read_index("data/index/faiss.bin")
        pass

    def encode_text(self, text: str):
        # TODO: return a 1-D numpy embedding vector for `text`
        raise NotImplementedError

    def encode_image(self, image_bytes: bytes):
        # TODO: return a 1-D numpy embedding vector for the image
        raise NotImplementedError

    def search(
        self,
        query: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        top_k: int = 10,
    ) -> List[SearchResult]:
        if self.faiss_index is None:
            return []

        # TODO: encode query, run FAISS search, return results
        # Example:
        #   import numpy as np
        #   if query:
        #       vec = self.encode_text(query)
        #   elif image_bytes:
        #       vec = self.encode_image(image_bytes)
        #   vec = vec.reshape(1, -1).astype(np.float32)
        #   distances, indices = self.faiss_index.search(vec, top_k)
        #   return [SearchResult(id=str(idx), score=float(1 - dist)) for idx, dist in zip(indices[0], distances[0])]
        return []
