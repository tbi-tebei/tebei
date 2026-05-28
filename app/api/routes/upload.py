import os
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.config import settings
from app.services.clip_service import clip_service
from app.services.data_store import data_store

router = APIRouter()

ALLOWED_EXT = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp"}


@router.post("/image")
async def upload_image(
    image: UploadFile = File(...),
    caption: str = Form(...),
):
    image_bytes = await image.read()
    if len(image_bytes) > settings.MAX_UPLOAD_SIZE:
        max_mb = settings.MAX_UPLOAD_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"File size exceeds the limit of {max_mb:.0f} MB",
        )

    ext = os.path.splitext(image.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    image_id = f"{uuid4().hex}{ext}"
    dest = os.path.join(settings.IMAGES_DIR, image_id)

    with open(dest, "wb") as f:
        f.write(image_bytes)

    clip_service.add_image(image_bytes, image_id)
    data_store.add_image(image_id, caption.strip())

    return {
        "image_id": image_id,
        "image_url": f"/images/{image_id}",
        "caption": caption.strip(),
    }
