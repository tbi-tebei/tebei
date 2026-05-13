import random
from app.services.data_store import data_store


class ImageRetriever:
    def search(self, image_bytes: bytes, top_k: int = 12) -> list[dict]:
        # MOCK — replace with CLIP or similar later
        ids = data_store.image_ids
        if not ids:
            return []
        k = min(top_k, len(ids))
        sampled = random.sample(ids, k)
        scores = sorted([random.uniform(0.4, 1.0) for _ in sampled], reverse=True)
        return [
            {
                "image_id": img,
                "score": round(score, 4),
                "caption": data_store.captions[img][0] if data_store.captions.get(img) else "",
            }
            for img, score in zip(sampled, scores)
        ]
