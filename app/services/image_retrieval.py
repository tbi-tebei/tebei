from app.services.data_store import data_store
from app.services.query_expansion import expand_image_query


class ImageRetriever:
    def search(self, image_bytes: bytes, top_k: int = 12) -> list[dict]:
        hits = expand_image_query(image_bytes, top_k)
        return [
            {
                "image_id": img_id,
                "score": round(score, 4),
                "caption": data_store.captions.get(img_id, [""])[0],
            }
            for img_id, score in hits
        ]
