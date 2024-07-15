from typing import Optional, Union
from llama_index.llms.azure_openai import AzureOpenAI

from pydantic import BaseModel
from llama_index.core.agent import FunctionCallingAgentWorker
from langchain_openai import AzureChatOpenAI

from web_scraping_agent.llm.base import llm_gpt4o
from web_scraping_agent.prompt.web_crawler import PMT_CRAWL_ALL
from web_scraping_agent.tools.web_parsing import li_scraper_tool, li_url_validation_tool
from web_scraping_agent.utils.scraping import scrape_firecrawl
from web_scraping_agent.agents.base import AgentBase
from llama_index.core.agent import ReActAgent


class WebCrawlerAgent(AgentBase):
    sys_prompt = PMT_CRAWL_ALL
    llm = llm_gpt4o
    tools = [li_scraper_tool, li_url_validation_tool]
