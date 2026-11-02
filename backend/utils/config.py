from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Groq (free!)
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-20b"
    embedding_backend: str = "local"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True
    secret_key: str = "dev-secret-change-me"

    # Database
    database_url: str = "sqlite:///./data/interview_coach.db"

    # FAISS persistence directory (reusing chroma key name for compatibility)
    chroma_persist_dir: str = "./data/vectordb"

    # Interview
    max_questions: int = 10

    # Frontend
    api_base_url: str = "http://localhost:8000"

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
