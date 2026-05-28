from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a .env file.

    All fields have sensible defaults so the application can run without a
    .env file during development.
    """

    APP_NAME: str = "Text-Image Search Engine"
    DATA_DIR: str = "data/raw"
    IMAGES_DIR: str = "data/raw/Images"
    CAPTIONS_FILE: str = "data/raw/captions.txt"
    INDEX_DIR: str = "data/index"
    MAX_UPLOAD_SIZE: int = 5 * 1024 * 1024
    GEMINI_API_KEY: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
