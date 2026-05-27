import requests
import numpy as np

from app.core.config import settings
from app.services.clip_service import clip_service

ALPHA = 1.0 # controls how much the original query influences the refined vector
BETA = 0.75 # controls how much the pseudo-relevant feedback shifts the query
GAMMA = 0.15 # controls how much the negative feedback pushes the query away
K_FEEDBACK = 10 # number of top results assumed relevant (positive examples)
K_NEGATIVE = 10 # number of bottom results used as negative examples

# Gemini configuration
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
    """Apply the full Rocchio formula with positive and negative feedback.

    Computes: refined = ALPHA * query + BETA * mean(positive) - GAMMA * mean(negative),
    then L2-normalizes so cosine similarity still works with IndexFlatIP.
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
    """Retrieve positive (top-K) and negative (bottom-K) indices.

    Positive: top-K most similar vectors (standard FAISS search).
    Negative: top-K most dissimilar vectors (search with negated query).
    """
    n_total = clip_service._index.ntotal
    k_pos = min(K_FEEDBACK, n_total)
    k_neg = min(K_NEGATIVE, n_total)

    emb_f32 = emb.astype(np.float32)
    _, pos_indices = clip_service._index.search(emb_f32, k_pos)
    positive = [int(i) for i in pos_indices[0] if i >= 0]

    _, neg_indices = clip_service._index.search(-emb_f32, k_neg)
    negative = [int(i) for i in neg_indices[0] if i >= 0]

    return positive, negative


def _call_gemini(query: str) -> list[str]:
    """Call Gemini API to generate visual descriptions for a query."""
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        return []

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
        return lines[:3]
    except Exception:
        return []


def _encode_with_llm(query: str) -> tuple[np.ndarray, list[str]]:
    """Encode query + LLM-generated descriptions into an averaged CLIP vector."""
    descriptions = _call_gemini(query)
    if not descriptions:
        return clip_service.encode_text(query), descriptions

    all_texts = [query] + descriptions
    embeddings = [clip_service.encode_text(t) for t in all_texts]
    avg_emb = np.mean(np.vstack(embeddings), axis=0, keepdims=True)
    avg_emb = avg_emb / np.linalg.norm(avg_emb)
    return avg_emb.astype(np.float32), descriptions


def expand_image_query(image_bytes: bytes, top_k: int = 12) -> list[tuple[str, float]]:
    """Image-to-image search with PRF only."""
    emb = clip_service.encode_image(image_bytes)
    if not clip_service.is_ready():
        return []

    positive, negative = _get_feedback(emb)
    if not positive:
        return clip_service.search_by_embedding(emb, top_k)

    refined = _rocchio_refine(emb, positive, negative)
    return clip_service.search_by_embedding(refined, top_k)


def expand_with_llm(query: str, top_k: int = 12) -> tuple[list[tuple[str, float]], list[str]]:
    """Text-to-image search with LLM expansion + PRF.

    1. Call Gemini to generate visual descriptions.
    2. Encode original query + descriptions with CLIP, average vectors.
    3. Run PRF (Rocchio with negative feedback) on the averaged vector.
    """
    avg_emb, descriptions = _encode_with_llm(query)
    if not clip_service.is_ready():
        return [], descriptions

    positive, negative = _get_feedback(avg_emb)
    if not positive:
        return clip_service.search_by_embedding(avg_emb, top_k), descriptions

    refined = _rocchio_refine(avg_emb, positive, negative)
    return clip_service.search_by_embedding(refined, top_k), descriptions
