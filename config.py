from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERP_API_KEY: str
    SEMANTIC_SCHOLAR_API_KEY: str

    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_GPT4O_MODEL: str
    AZURE_OPENAI_EMBEDDING_MODEL: str
    MAX_TOKENS: int

    AZURE_DYNAMIC_SESSION_MGMT_ENDPOINT: str

    LLAMA_CLOUD_API_KEY: str

    # vector store
    QDRANT_HOST: str
    QDRANT_PORT: str

    # doc store
    REDIS_HOST: str
    REDIS_PORT: int

    class Config:
        env_file = ".env"


settings = Settings()
