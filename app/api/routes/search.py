from fastapi import APIRouter, Request, UploadFile, File, Form, Query
from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services.text_retrieval import TextRetriever
from app.services.image_retrieval import ImageRetriever
from app.services.data_store import data_store
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)
text_retriever = TextRetriever()
image_retriever = ImageRetriever()


@router.post("/text", response_model=SearchResponse)
@limiter.limit("15/minute")
async def text_search(request: Request, search_request: SearchRequest):
    hits = text_retriever.search(search_request.query, top_k=search_request.top_k)
    results = [
        SearchResult(
            image_id=h["image_id"],
            image_url=f"/images/{h['image_id']}",
            score=h["score"],
            caption=h["caption"],
        )
        for h in hits
    ]
    return SearchResponse(query=search_request.query, results=results, total=len(results))


@router.post("/image", response_model=SearchResponse)
@limiter.limit("10/minute")
async def image_search(
    request: Request,
    image: UploadFile = File(...),
    top_k: int = Form(12),
):
    image_bytes = await image.read()
    hits = image_retriever.search(image_bytes, top_k=top_k)
    results = [
        SearchResult(
            image_id=h["image_id"],
            image_url=f"/images/{h['image_id']}",
            score=h["score"],
            caption=h["caption"],
        )
        for h in hits
    ]
    return SearchResponse(query="image similarity search", results=results, total=len(results))


@router.get("/suggestions")
async def get_suggestions():
    return {"suggestions": data_store.suggestions}


@router.get("/autocomplete")
async def autocomplete(q: str = Query("", min_length=1)):
    return {"results": data_store.autocomplete(q)}
