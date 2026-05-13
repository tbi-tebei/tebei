from fastapi import APIRouter
from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.services.text_retrieval import TextRetriever

router = APIRouter()
retriever = TextRetriever()


@router.post("/text", response_model=SearchResponse)
async def text_search(request: SearchRequest):
    hits = retriever.search(request.query, top_k=request.top_k)
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
