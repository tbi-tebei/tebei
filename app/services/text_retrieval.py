from app.services.data_store import data_store
from app.services.query_expansion import expand_text_query
from app.services.reranker import reranker


class TextRetriever:
    def search(self, query: str, top_k: int = 12) -> list[dict]:
        hits = expand_text_query(query, top_k=top_k * 3)
        reranked_hits = reranker.rerank(
            query=query,
            candidates=hits,
            captions=data_store.captions,
            top_k=top_k
        )
        return [
            {
                "image_id": img_id,
                "score": round(score, 4),
                "caption": data_store.captions.get(img_id, [""])[0],
            }
            for img_id, score in reranked_hits
        ]
