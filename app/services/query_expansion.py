from functools import lru_cache

import requests
import numpy as np

from app.core.config import settings
from app.services.clip_service import clip_service

# Rocchio feedback weights
ALPHA = 1.0       # weight for the original query vector
BETA = 0.75       # weight for the positive-example centroid
GAMMA = 0.15      # weight for the negative-example centroid
K_FEEDBACK = 10   # number of top results used as positive pseudo-relevant examples
K_NEGATIVE = 10   # number of bottom results used as negative examples

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
    f":generateContent?key={{}}"
)
PROMPT_TEMPLATE = (
    'For the query "{query}", write 3 photo descriptions, one per line.\n'
    "Each line is a short sentence describing what the photo looks like.\n"
    "No numbering, no explanation, just 3 lines."
)


def _rocchio_refine(
    query_emb: np.ndarray,
    positive_indices: list[int],
    negative_indices: list[int],
) -> np.ndarray:
    """Apply the Rocchio algorithm to refine a query embedding.

    Formula:
        refined = ALPHA * q + BETA * mean(positive) - GAMMA * mean(negative)

    The result is L2-normalised so it remains compatible with the
    inner-product similarity used by ``IndexFlatIP``.
    """
    pos_vecs = clip_service.get_vectors(positive_indices)
    pos_centroid = np.mean(pos_vecs, axis=0, keepdims=True)

    refined = ALPHA * query_emb + BETA * pos_centroid

    if negative_indices:
        neg_vecs = clip_service.get_vectors(negative_indices)
        neg_centroid = np.mean(neg_vecs, axis=0, keepdims=True)
        refined = refined - GAMMA * neg_centroid

    norm = np.linalg.norm(refined)
    if norm > 0:
        refined = refined / norm
    return refined.astype(np.float32)


def _get_feedback(emb: np.ndarray) -> tuple[list[int], list[int]]:
    """Return positive and negative pseudo-relevant document indices.

    Positive indices are the top-K most similar vectors found via FAISS search.
    Negative indices are the top-K most dissimilar vectors found via brute-force
    dot product over the raw embedding matrix.
    """
    n_total = clip_service._index.ntotal
    k_pos = min(K_FEEDBACK, n_total)
    k_neg = min(K_NEGATIVE, n_total)

    emb_f32 = emb.astype(np.float32)
    _, pos_indices = clip_service._index.search(emb_f32, k_pos)
    positive = [int(i) for i in pos_indices[0] if i >= 0]
    negative = clip_service.search_negative(emb_f32, k_neg)

    return positive, negative


@lru_cache(maxsize=256)
def _call_gemini(query: str) -> tuple[str, ...]:
    """Call the Gemini API to generate visual photo descriptions for a query.

    Results are cached by query string to avoid redundant API calls for
    repeated or popular searches. Returns an empty tuple when the API key is
    missing or the request fails, so callers can fall back gracefully.
    """
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        return ()

    url = GEMINI_URL.format(api_key)
    payload = {
        "contents": [{"parts": [{"text": PROMPT_TEMPLATE.format(query=query)}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 1024,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        return tuple(lines[:3])
    except Exception:
        return ()


def _encode_with_llm(query: str) -> tuple[np.ndarray, list[str]]:
    """Encode a query enriched with LLM-generated visual descriptions.

    Calls Gemini to produce up to 3 photo descriptions, then encodes the
    original query together with those descriptions using CLIP and returns
    their L2-normalised average as a single query vector. Falls back to
    encoding the raw query alone when Gemini is unavailable.
    """
    descriptions = list(_call_gemini(query))
    if not descriptions:
        return clip_service.encode_text(query), descriptions

    all_texts = [query] + descriptions
    embeddings = clip_service.encode_texts(all_texts)
    avg_emb = np.mean(embeddings, axis=0, keepdims=True)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)
    return avg_emb.astype(np.float32), descriptions


def expand_image_query(image_bytes: bytes, top_k: int = 12) -> list[tuple[str, float]]:
    """Run image-to-image search with Rocchio pseudo-relevance feedback.

    Encodes the query image with CLIP, refines the embedding using Rocchio
    PRF, then returns the top-k most similar images from the FAISS index.
    Falls back to a plain embedding search when feedback data is unavailable.
    """
    emb = clip_service.encode_image(image_bytes)
    if not clip_service.is_ready():
        return []

    positive, negative = _get_feedback(emb)
    if not positive:
        return clip_service.search_by_embedding(emb, top_k)

    refined = _rocchio_refine(emb, positive, negative)
    return clip_service.search_by_embedding(refined, top_k)


def expand_with_llm(query: str, top_k: int = 12) -> tuple[list[tuple[str, float]], list[str]]:
    """Run text-to-image search with LLM expansion and Rocchio PRF.

    Pipeline:
    1. Call Gemini to generate visual descriptions for the query.
    2. Encode query + descriptions with CLIP and average the vectors.
    3. Refine the averaged vector with Rocchio PRF (positive + negative feedback).
    4. Search the FAISS index with the refined vector.

    Returns a tuple of (ranked results, llm descriptions used).
    """
    avg_emb, descriptions = _encode_with_llm(query)
    if not clip_service.is_ready():
        return [], descriptions

    positive, negative = _get_feedback(avg_emb)
    if not positive:
        return clip_service.search_by_embedding(avg_emb, top_k), descriptions

    refined = _rocchio_refine(avg_emb, positive, negative)
    return clip_service.search_by_embedding(refined, top_k), descriptions
