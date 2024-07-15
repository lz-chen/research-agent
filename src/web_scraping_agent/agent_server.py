# https://mer.vin/2024/06/llama-agents-example/

from llama_agents import (
    AgentService,
    ControlPlaneServer,
    SimpleMessageQueue,
    PipelineOrchestrator,
    ServiceComponent,
    LocalLauncher,
)
from llama_agents.tools import AgentServiceTool
from llama_index.core.agent import ReActAgent
from llama_index.agent.openai import OpenAIAgent

from web_scraping_agent.agents.report_gen import ReportGenAgent
from web_scraping_agent.agents.webpage_crawler import WebCrawlerAgent
from llama_index.core.query_pipeline import QueryPipeline

from web_scraping_agent.llm.base import llm_gpt4o
from llama_index.core import Settings

Settings.llm = llm_gpt4o

message_queue = SimpleMessageQueue()

# web crawler agent
web_crawler_server = AgentService(
    agent=WebCrawlerAgent().setup(),
    message_queue=message_queue,
    description="Useful agent for scraping web page of company for information",
    service_name="web_crawler_agent",
)
web_crawler_server_tool = AgentServiceTool.from_service_definition(
    message_queue=message_queue,
    service_definition=web_crawler_server.service_definition
)

# report gen agent
report_gen_server = AgentService(
    agent=ReportGenAgent().setup(),
    message_queue=message_queue,
    description="Useful for generating a report based on the information scraped from a company's web page",
    service_name="report_gen_agent",
)
report_gen_server_tool = AgentServiceTool.from_service_definition(
    message_queue=message_queue,
    service_definition=report_gen_server.service_definition
)

# main manager agent, using other two agents as tool
manager_agent = OpenAIAgent.from_tools(
    [web_crawler_server_tool, report_gen_server_tool],
    system_prompt="Gather information about a company and generate a report "
                  "based on the url that user provided.",
)
# manager_agent = ReActAgent.from_tools([web_crawler_server_tool, report_gen_server_tool],
#                                       llm=llm_gpt4o, verbose=True)

manager_agent_server = AgentService(
    agent=manager_agent,
    message_queue=message_queue,
    description="Manager agent that orchestrates company information gathering and "
                "report generation.",
    service_name="manager_agent",
)

manager_agent_component = ServiceComponent.from_service_definition(manager_agent_server.service_definition)

pipeline = QueryPipeline(chain=[manager_agent_component])

pipeline_orchestrator = PipelineOrchestrator(pipeline)

control_plane = ControlPlaneServer(message_queue, pipeline_orchestrator)

# launch it
launcher = LocalLauncher([web_crawler_server, report_gen_server, manager_agent_server],
                         control_plane,
                         message_queue)
result = launcher.launch_single(
    "Get me the overall information of the company from this page https://presail.com/")

print(f"Result: {result}")
