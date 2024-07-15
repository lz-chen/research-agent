from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_GPT4O_MODEL: str
    AZURE_OPENAI_GPT4_MODEL: str
    AZURE_OPENAI_API_VERSION: str

    # either use jina or firecrawl
    CRAWLER: Literal['jina', 'firecrawl']
    FIRECRAWL_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()
