from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import search, index
from app.core.config import settings

app = FastAPI(
    title=settings.APP_NAME,
    description="Multimodal text-image search engine — IR course final project",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(index.router, prefix="/api/index", tags=["index"])

templates = Jinja2Templates(directory="app/templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "app_name": settings.APP_NAME})


@app.get("/health")
async def health():
    return {"status": "ok"}
