from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SERP_API_KEY: str
    SEMANTIC_SCHOLAR_API_KEY: str
    TAVILY_API_KEY: str

    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_GPT4O_MODEL: str
    AZURE_OPENAI_GPT4O_MINI_MODEL: str
    AZURE_OPENAI_EMBEDDING_MODEL: str
    MAX_TOKENS: int

    AZURE_DYNAMIC_SESSION_MGMT_ENDPOINT: str

    # vector store
    QDRANT_HOST: str
    QDRANT_PORT: str

    # doc store
    REDIS_HOST: str
    REDIS_PORT: int

    WORKFLOW_ARTIFACTS_PATH: str

    MLFLOW_TRACKING_URI: str

    class Config:
        env_file = ".env"  # relative to execution path


settings = Settings()
