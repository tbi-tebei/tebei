from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Text-Image Search Engine"
    DATA_DIR: str = "data/raw"
    INDEX_DIR: str = "data/index"

    # Retrieval defaults
    TOP_K: int = 10

    # Model config (for CLIP or sentence-transformers)
    CLIP_MODEL: str = "ViT-B/32"
    EMBEDDING_DIM: int = 512

    class Config:
        env_file = ".env"


settings = Settings()
