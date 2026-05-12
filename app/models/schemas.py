from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


class SearchMode(str, Enum):
    text = "text"
    image = "image"
    multimodal = "multimodal"


class TextSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    mode: SearchMode = SearchMode.text


class SearchResult(BaseModel):
    id: str
    score: float
    title: Optional[str] = None
    text: Optional[str] = None
    image_path: Optional[str] = None
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int
    took_ms: float


class IndexStatusResponse(BaseModel):
    ready: bool
    docs_indexed: int
    index_path: Optional[str] = None
