import numpy as np

from app.services.clip_service import clip_service

ALPHA = 1.0 # controls how much the original query influences the refined vector
BETA = 0.75 # controls how much the pseudo-relevant feedback shifts the query
K_FEEDBACK = 10


def _rocchio_refine(query_emb: np.ndarray, feedback_indices: list[int]) -> np.ndarray:
    """Apply the Rocchio formula to produce a refined query vector.

    Computes: refined = ALPHA * query + BETA * mean(feedback vectors),
    then L2-normalizes so cosine similarity still works with IndexFlatIP.
    """
    feedback_vecs = clip_service.get_vectors(feedback_indices)
    centroid = np.mean(feedback_vecs, axis=0, keepdims=True)
    refined = ALPHA * query_emb + BETA * centroid
    norm = np.linalg.norm(refined)
    if norm > 0:
        refined = refined / norm
    return refined.astype(np.float32)


def expand_text_query(query: str, top_k: int = 12) -> list[tuple[str, float]]:
    """Run text-to-image search with pseudo-relevance feedback.

    1. Encode text query with CLIP.
    2. Retrieve top-K_FEEDBACK images from FAISS (initial pass).
    3. Refine the query vector using Rocchio.
    4. Retrieve final top_k images with the refined vector (second pass).
    """
    emb = clip_service.encode_text(query)
    if not clip_service.is_ready():
        return []

    k_fb = min(K_FEEDBACK, clip_service._index.ntotal)
    scores, indices = clip_service._index.search(emb.astype(np.float32), k_fb)
    feedback_indices = [i for i in indices[0] if i >= 0]

    if not feedback_indices:
        return clip_service.search_by_embedding(emb, top_k)

    refined = _rocchio_refine(emb, feedback_indices)
    return clip_service.search_by_embedding(refined, top_k)


def expand_image_query(image_bytes: bytes, top_k: int = 12) -> list[tuple[str, float]]:
    """Run image-to-image search with pseudo-relevance feedback.

    Same two-pass Rocchio approach as text, but the initial query vector
    comes from encoding the uploaded image with CLIP instead of text.
    """
    emb = clip_service.encode_image(image_bytes)
    if not clip_service.is_ready():
        return []

    k_fb = min(K_FEEDBACK, clip_service._index.ntotal)
    scores, indices = clip_service._index.search(emb.astype(np.float32), k_fb)
    feedback_indices = [i for i in indices[0] if i >= 0]

    if not feedback_indices:
        return clip_service.search_by_embedding(emb, top_k)

    refined = _rocchio_refine(emb, feedback_indices)
    return clip_service.search_by_embedding(refined, top_k)
