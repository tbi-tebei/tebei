import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.config import settings
from app.services.data_store import data_store

router = APIRouter()

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}


@router.post("/image")
async def upload_image(
    image: UploadFile = File(...),
    caption: str = Form(...),
):
    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    dest = os.path.join(settings.IMAGES_DIR, image.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(image.file, f)

    data_store.add_image(image.filename, caption.strip())

    return {
        "image_id": image.filename,
        "image_url": f"/images/{image.filename}",
        "caption": caption.strip(),
    }
