#!/usr/bin/env python3
"""Encode all dataset images with CLIP and build a FAISS index.

Run from the project root:
    source env/bin/activate
    python scripts/build_index.py
"""
import json
import os
import sys

import faiss
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer

from app.core.config import settings

IMAGE_MODEL_NAME = "clip-ViT-B-32"  # image encoder (standard CLIP)
BATCH_SIZE = 64


def main():
    print(f"Model : {IMAGE_MODEL_NAME}")
    model = SentenceTransformer(IMAGE_MODEL_NAME, device="cpu")

    images_dir = settings.IMAGES_DIR
    index_dir = settings.INDEX_DIR
    os.makedirs(index_dir, exist_ok=True)

    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"Images: {len(image_files)} found in {images_dir}")

    all_embeddings: list[np.ndarray] = []
    all_ids: list[str] = []

    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Encoding"):
        batch_files = image_files[i: i + BATCH_SIZE]
        imgs, ids = [], []
        for fname in batch_files:
            try:
                img = Image.open(os.path.join(images_dir, fname)).convert("RGB")
                imgs.append(img)
                ids.append(fname)
            except Exception as e:
                print(f"  skip {fname}: {e}")
        if imgs:
            embs = model.encode(imgs, convert_to_numpy=True, normalize_embeddings=True)
            all_embeddings.append(embs)
            all_ids.extend(ids)

    matrix = np.vstack(all_embeddings).astype(np.float32)
    print(f"Matrix : {matrix.shape}")

    # IndexFlatIP = cosine similarity (vectors are L2-normalized)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    index_path = os.path.join(index_dir, "image_index.faiss")
    ids_path = os.path.join(index_dir, "image_ids.json")
    faiss.write_index(index, index_path)
    with open(ids_path, "w") as f:
        json.dump(all_ids, f)

    print(f"Saved : {index.ntotal} vectors → {index_path}")
    print(f"IDs   : {ids_path}")


if __name__ == "__main__":
    main()
