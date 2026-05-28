from pydantic import BaseModel
from typing import List


class SearchRequest(BaseModel):
    """Payload for a text-to-image search request."""

    query: str
    top_k: int = 12


class SearchResult(BaseModel):
    """A single ranked result returned by the search pipeline."""

    image_id: str
    image_url: str
    score: float
    caption: str


class SearchResponse(BaseModel):
    """Full response envelope for both text and image search endpoints."""

    query: str
    results: List[SearchResult]
    total: int
