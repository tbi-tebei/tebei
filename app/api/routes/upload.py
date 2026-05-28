import os
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

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
    """Upload a new image with a caption and insert it into the live index.

    The image is encoded with CLIP and added to the FAISS index immediately,
    making it searchable without a server restart. The caption is appended to
    ``captions.txt`` for persistence across restarts.

    Raises:
        413: if the file exceeds ``MAX_UPLOAD_SIZE`` (default 5 MB).
        400: if the file extension is not in ``ALLOWED_EXT``.
    """
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
