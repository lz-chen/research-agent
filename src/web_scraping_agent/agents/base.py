from typing import Optional, Union
from llama_index.llms.azure_openai import AzureOpenAI

from pydantic import BaseModel, Field
from llama_index.core.agent import FunctionCallingAgentWorker
from langchain_openai import AzureChatOpenAI
from pydantic.dataclasses import dataclass


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class AgentBase:
    llm: Optional[Union[AzureOpenAI, AzureChatOpenAI]] = Field(None, arbitrary_types_allowed=True)
    tools: Optional[list] = None
    sys_prompt: Optional[str] = None

    def setup(self):
        worker = FunctionCallingAgentWorker.from_tools(self.tools,
                                                       llm=self.llm,
                                                       system_prompt=self.sys_prompt)
        return worker.as_agent()
