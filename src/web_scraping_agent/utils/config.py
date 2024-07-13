from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_CHAT_MODEL: str
    AZURE_OPENAI_API_VERSION: str

    FIRECRAWL_API_KEY: str


settings = Settings()
