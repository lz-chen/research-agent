from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal

from config import settings

llm_gpt4o = AzureOpenAI(
    azure_deployment=settings.AZURE_OPENAI_GPT4O_MODEL,
    model=settings.AZURE_OPENAI_GPT4O_MODEL,  # this name will be used in trace
    temperature=0.0,
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)

mm_gpt4o = AzureOpenAIMultiModal(
    azure_deployment=settings.AZURE_OPENAI_GPT4O_MODEL,
    temperature=0.0,
    max_new_tokens=4096,  # 4096 is the maximum number of tokens allowed by the API
    azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
    api_key=settings.AZURE_OPENAI_API_KEY,
    api_version=settings.AZURE_OPENAI_API_VERSION,
)


def new_gpt4o(temperature=0.0):
    return AzureOpenAI(
        azure_deployment=settings.AZURE_OPENAI_GPT4O_MODEL,
        model=settings.AZURE_OPENAI_GPT4O_MODEL,  # this name will be used in trace
        temperature=temperature,
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
    )
