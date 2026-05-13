from pydantic import BaseModel
from typing import List


class SearchRequest(BaseModel):
    query: str
    top_k: int = 12


class SearchResult(BaseModel):
    image_id: str
    image_url: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
