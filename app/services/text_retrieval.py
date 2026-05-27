from app.services.data_store import data_store
from app.services.query_expansion import expand_text_query
from app.services.reranker import reranker
from deep_translator import GoogleTranslator


RERANKER_FALLBACK_THRESHOLD = 0.1

# Translate query to English for better reranking performance
def _translate_to_english(query: str) -> str:
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(query)
        return translated if translated else query
    except Exception:
        return query

class TextRetriever:
    def search(self, query: str, top_k: int = 12) -> list[dict]:
        candidates = expand_text_query(query, top_k=top_k * 3)

        translated_query = _translate_to_english(query)
        reranked_hits = reranker.rerank(
            query=translated_query,
            candidates=candidates,
            captions=data_store.captions,
            top_k=top_k
        )
        # Fallback to CLIP scores if reranker scores are too low
        top_score = reranked_hits[0][1] if reranked_hits else 0
        if top_score < RERANKER_FALLBACK_THRESHOLD:
            hits = expand_text_query(query, top_k=top_k)
            return [
                {
                    "image_id": img_id,
                    "score": round(score, 4),
                    "caption": data_store.captions.get(img_id, [""])[0],
                }
                for img_id, score in hits
            ]
        
        return [
            {
                "image_id": img_id,
                "score": round(score, 4),
                "caption": data_store.captions.get(img_id, [""])[0],
            }
            for img_id, score in reranked_hits
        ]
