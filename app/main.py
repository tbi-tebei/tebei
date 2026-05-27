import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import search, upload
from app.core.config import settings

logger = logging.getLogger(__name__)

os.makedirs(settings.IMAGES_DIR, exist_ok=True)
os.makedirs(settings.INDEX_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.clip_service import clip_service
    from app.services.reranker import reranker

    logger.info("Loading CLIP model and FAISS index...")
    clip_service.is_ready()
    logger.info("Loading cross-encoder reranker...")
    reranker._load()
    logger.info("Models loaded.")
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/public", StaticFiles(directory="app/public"), name="public")
app.mount("/images", StaticFiles(directory=settings.IMAGES_DIR), name="images")

app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])

templates = Jinja2Templates(directory="app/templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    return {"status": "ok"}
