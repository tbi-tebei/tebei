from fastapi import APIRouter, HTTPException
from app.models.schemas import IndexStatusResponse
from app.services.indexer import Indexer

router = APIRouter()

indexer = Indexer()


@router.post("/build")
async def build_index():
    try:
        result = indexer.build()
        return {"message": "Index built successfully", **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=IndexStatusResponse)
async def index_status():
    return indexer.status()
