from fastapi import APIRouter, UploadFile, File, Form
from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services.text_retrieval import TextRetriever
from app.services.image_retrieval import ImageRetriever

router = APIRouter()
text_retriever = TextRetriever()
image_retriever = ImageRetriever()


@router.post("/text", response_model=SearchResponse)
async def text_search(request: SearchRequest):
    hits = text_retriever.search(request.query, top_k=request.top_k)
    results = [
        SearchResult(
            image_id=h["image_id"],
            image_url=f"/images/{h['image_id']}",
            score=h["score"],
            caption=h["caption"],
        )
        for h in hits
    ]
    return SearchResponse(query=request.query, results=results, total=len(results))


@router.post("/image", response_model=SearchResponse)
async def image_search(
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
