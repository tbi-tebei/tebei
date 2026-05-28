import logging
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.api.routes import search, upload
from app.core.config import settings

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
logger = logging.getLogger(__name__)

os.makedirs(settings.IMAGES_DIR, exist_ok=True)
os.makedirs(settings.INDEX_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CLIP model and FAISS index once at startup so the first request
    does not pay the cold-start cost."""
    from app.services.clip_service import clip_service

    logger.info("Loading CLIP model and FAISS index...")
    clip_service.is_ready()
    logger.info("Models loaded.")
    yield


app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/public", StaticFiles(directory="app/public"), name="public")
app.mount("/images", StaticFiles(directory=settings.IMAGES_DIR), name="images")

app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])

templates = Jinja2Templates(directory="app/templates")


@app.get("/")
async def home(request: Request):
    """Serve the single-page search UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health")
async def health():
    """Liveness probe used by Docker health checks and uptime monitors."""
    return {"status": "ok"}
