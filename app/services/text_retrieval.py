from app.services.clip_service import clip_service
from app.services.data_store import data_store


class TextRetriever:
    def search(self, query: str, top_k: int = 12) -> list[dict]:
        hits = clip_service.search_by_text(query, top_k)
        return [
            {
                "image_id": img_id,
                "score": round(score, 4),
                "caption": data_store.captions.get(img_id, [""])[0],
            }
            for img_id, score in hits
        ]
