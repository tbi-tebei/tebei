from fastapi import APIRouter
from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services.text_retrieval import TextRetriever

router = APIRouter()
retriever = TextRetriever()


@router.post("/text", response_model=SearchResponse)
async def text_search(request: SearchRequest):
    image_ids = retriever.search(request.query, top_k=request.top_k)
    results = [
        SearchResult(image_id=img, image_url=f"/images/{img}")
        for img in image_ids
    ]
    return SearchResponse(query=request.query, results=results, total=len(results))
