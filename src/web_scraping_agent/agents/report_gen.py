from llama_index.core.agent import FunctionCallingAgentWorker

from web_scraping_agent.agents.base import AgentBase
from web_scraping_agent.llm.base import llm_gpt4o
from web_scraping_agent.prompt.report_gen import PMT_COMP_REPORT


class ReportGenAgent(AgentBase):
    sys_prompt = PMT_COMP_REPORT
    llm = llm_gpt4o
    tools = []
