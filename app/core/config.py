from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Text-Image Search Engine"
    DATA_DIR: str = "data/raw"
    IMAGES_DIR: str = "data/raw/Images"
    CAPTIONS_FILE: str = "data/raw/captions.txt"
    INDEX_DIR: str = "data/index"
    GEMINI_API_KEY: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
