from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_PATH) if _ENV_PATH.exists() else ".env",
        extra="ignore",
        env_file_encoding="utf-8",
    )

    llm_provider: str = "local"
    embedding_provider: str = "local"
    vector_store_provider: str = "memory"
    openai_base_url: str = "http://localhost:1234/v1"
    openai_api_key: str = "lm-studio"
    llm_model: str = "qwen"
    embedding_model: str = "text-embedding-model"
    embedding_dimension: int = 256
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_path: str = ""  # embedded local Qdrant (pure Python, no server/Docker)
    collection_name: str = "vishnu_rag"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    max_candidates: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
