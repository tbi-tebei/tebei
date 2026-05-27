import csv
import os
from app.core.config import settings


class DataStore:
    """Single in-memory store for image IDs and captions, shared across services."""

    def __init__(self):
        self.image_ids: list[str] = []
        self.captions: dict[str, list[str]] = {}
        self.load()

    def load(self):
        self.image_ids.clear()
        self.captions.clear()
        path = settings.CAPTIONS_FILE
        if not os.path.exists(path):
            return
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if not row:
                    continue
                img = row[0].strip()
                caption = row[1].strip() if len(row) > 1 else ""
                if not img:
                    continue
                if img not in self.captions:
                    self.captions[img] = []
                    self.image_ids.append(img)
                self.captions[img].append(caption)

    def add_image(self, image_id: str, caption: str):
        if image_id not in self.captions:
            self.captions[image_id] = []
            self.image_ids.append(image_id)
        self.captions[image_id].append(caption)
        
        # Ensure file ends with newline before appending
        if os.path.exists(settings.CAPTIONS_FILE):
            with open(settings.CAPTIONS_FILE, "rb") as f:
                f.seek(0, 2)
                if f.tell() > 0:
                    f.seek(-1, 2)
                    if f.read(1) != b"\n":
                        with open(settings.CAPTIONS_FILE, "a", encoding="utf-8") as nf:
                            nf.write("\n")
        
        with open(settings.CAPTIONS_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([image_id, caption])


data_store = DataStore()
