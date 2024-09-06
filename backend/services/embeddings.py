from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from config import settings

aoai_embedder = AzureOpenAIEmbedding(
    model=settings.AZURE_OPENAI_EMBEDDING_MODEL,
    # deployment_name=settings.AZURE_OPENAI_TEXT_EMBEDDING_MODEL,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_version=settings.AZURE_OPENAI_API_VERSION,
    api_key=settings.AZURE_OPENAI_API_KEY,
)
