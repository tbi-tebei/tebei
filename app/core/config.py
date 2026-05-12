from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Text-Image Search Engine"

    class Config:
        env_file = ".env"


settings = Settings()
