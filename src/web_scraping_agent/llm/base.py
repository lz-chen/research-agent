from llama_index.llms.azure_openai import AzureOpenAI, AsyncAzureOpenAI
from web_scraping_agent.utils.config import settings

llm_gpt4o = AzureOpenAI(
    azure_deployment=settings.AZURE_OPENAI_GPT4O_MODEL,
    temperature=0.,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

allm_gpt4o = AsyncAzureOpenAI(
    azure_deployment=settings.AZURE_OPENAI_GPT4O_MODEL,
    # temperature=0.,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)