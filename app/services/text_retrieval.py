from app.services.data_store import data_store
from app.services.query_expansion import expand_with_llm
from app.services.reranker import reranker
from deep_translator import GoogleTranslator


RERANKER_FALLBACK_THRESHOLD = 0.1


def _translate_to_english(query: str) -> str:
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(query)
        return translated if translated else query
    except Exception:
        return query


def _fuse_results(
    vector_hits: list[tuple[str, float]],
    bm25_hits: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Combine vector and BM25 results using reciprocal rank fusion (RRF)."""
    k = 60
    scores: dict[str, float] = {}
    for rank, (img_id, _) in enumerate(vector_hits):
        scores[img_id] = scores.get(img_id, 0) + 1.0 / (k + rank + 1)
    for rank, (img_id, _) in enumerate(bm25_hits):
        scores[img_id] = scores.get(img_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _format_hits(hits):
    return [
        {
            "image_id": img_id,
            "score": round(score, 4),
            "caption": data_store.captions.get(img_id, [""])[0],
        }
        for img_id, score in hits
    ]


class TextRetriever:
    def search(self, query: str, top_k: int = 12) -> list[dict]:
        translated_query = _translate_to_english(query)

        vector_hits, _ = expand_with_llm(translated_query, top_k=top_k * 3)
        bm25_hits = data_store.search_bm25(translated_query, top_k=top_k * 3)

        fused = _fuse_results(vector_hits, bm25_hits)

        reranked_hits = reranker.rerank(
            query=translated_query,
            candidates=fused,
            captions=data_store.captions,
            top_k=top_k,
        )

        top_score = reranked_hits[0][1] if reranked_hits else 0
        if top_score < RERANKER_FALLBACK_THRESHOLD:
            return _format_hits(fused[:top_k])

        return _format_hits(reranked_hits)
