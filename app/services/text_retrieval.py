from app.services.data_store import data_store
from app.services.query_expansion import expand_with_llm
from app.services.reranker import reranker
from deep_translator import GoogleTranslator

RERANKER_FALLBACK_THRESHOLD = 0.1


def _translate_to_english(query: str) -> str:
    """Auto-detect the query language and translate it to English.

    Returns the original query unchanged if translation fails, so the
    pipeline degrades gracefully without raising an exception.
    """
    try:
        translated = GoogleTranslator(source="auto", target="en").translate(query)
        return translated if translated else query
    except Exception:
        return query


def _fuse_results(
    vector_hits: list[tuple[str, float]],
    bm25_hits: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Merge semantic and lexical result lists using Reciprocal Rank Fusion.

    RRF score: score(d) = sum(1 / (k + rank_i(d))) where k=60.
    Documents appearing in both lists receive a higher combined score than
    those found by only one retriever.
    """
    k = 60
    scores: dict[str, float] = {}
    for rank, (img_id, _) in enumerate(vector_hits):
        scores[img_id] = scores.get(img_id, 0) + 1.0 / (k + rank + 1)
    for rank, (img_id, _) in enumerate(bm25_hits):
        scores[img_id] = scores.get(img_id, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _format_hits(hits: list[tuple[str, float]]) -> list[dict]:
    """Convert (image_id, score) pairs into the response dict format."""
    return [
        {
            "image_id": img_id,
            "score": round(score, 4),
            "caption": data_store.captions.get(img_id, [""])[0],
        }
        for img_id, score in hits
    ]


class TextRetriever:
    """End-to-end text-to-image retrieval pipeline.

    Stages (in order):
    1. Translate the query to English.
    2. Expand the query with LLM-generated visual descriptions + Rocchio PRF.
    3. Run parallel semantic (CLIP + FAISS) and lexical (BM25) retrieval.
    4. Fuse the two result lists with Reciprocal Rank Fusion.
    5. Rerank the fused candidates with a cross-encoder.
    """

    def search(self, query: str, top_k: int = 12) -> list[dict]:
        """Execute the full retrieval pipeline and return ranked results.

        Falls back to the RRF-fused ranking when the cross-encoder top score
        is below ``RERANKER_FALLBACK_THRESHOLD``, which indicates that no
        (query, caption) pair is a strong match and reranking would degrade
        the result quality.
        """
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
