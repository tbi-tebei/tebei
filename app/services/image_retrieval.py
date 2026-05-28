from app.services.data_store import data_store
from app.services.query_expansion import expand_image_query


class ImageRetriever:
    """Image-to-image retrieval pipeline.

    Encodes the query image with CLIP, refines the embedding with Rocchio
    pseudo-relevance feedback, and searches the FAISS index for visually
    similar images.
    """

    def search(self, image_bytes: bytes, top_k: int = 12) -> list[dict]:
        """Return the top-k images most visually similar to the query image."""
        hits = expand_image_query(image_bytes, top_k)
        return [
            {
                "image_id": img_id,
                "score": round(score, 4),
                "caption": data_store.captions.get(img_id, [""])[0],
            }
            for img_id, score in hits
        ]
