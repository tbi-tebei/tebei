import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional

from app.models.schemas import TextSearchRequest, SearchResponse
from app.services.text_retrieval import TextRetriever
from app.services.image_retrieval import ImageRetriever

router = APIRouter()

text_retriever = TextRetriever()
image_retriever = ImageRetriever()


@router.post("/text", response_model=SearchResponse)
async def text_search(request: TextSearchRequest):
    start = time.time()
    results = text_retriever.search(request.query, top_k=request.top_k)
    took_ms = (time.time() - start) * 1000
    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
        took_ms=round(took_ms, 2),
    )


@router.post("/image", response_model=SearchResponse)
async def image_search(
    query: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
):
    if not query and not image:
        raise HTTPException(status_code=400, detail="Provide a text query or an image file")

    start = time.time()
    image_bytes = await image.read() if image else None
    results = image_retriever.search(query=query, image_bytes=image_bytes, top_k=top_k)
    took_ms = (time.time() - start) * 1000
    return SearchResponse(
        query=query or "image query",
        results=results,
        total=len(results),
        took_ms=round(took_ms, 2),
    )
