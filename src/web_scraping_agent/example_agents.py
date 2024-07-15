from llama_agents import (
    AgentService,
    ControlPlaneServer,
    SimpleMessageQueue,
    PipelineOrchestrator,
    ServiceComponent,
    LocalLauncher,
)
from llama_agents.tools import AgentServiceTool

from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.settings import Settings

from web_scraping_agent.llm.base import llm_gpt4o
from llama_index.core.agent import ReActAgent

Settings.llm = llm_gpt4o


# create an agent
def get_the_secret_fact() -> str:
    """Returns the secret fact."""
    return "The secret fact is: A baby llama is called a 'Cria'."


tool = FunctionTool.from_defaults(fn=get_the_secret_fact)

worker1 = FunctionCallingAgentWorker.from_tools([tool], llm=OpenAI())
# worker2 = FunctionCallingAgentWorker.from_tools([], llm=OpenAI())
agent1 = worker1.as_agent()

# create our multi-agent framework components
message_queue = SimpleMessageQueue()

agent1_server = AgentService(
    agent=agent1,
    message_queue=message_queue,
    description="Useful for getting the secret fact.",
    service_name="secret_fact_agent",
)

agent1_server_tool = AgentServiceTool.from_service_definition(
    message_queue=message_queue, service_definition=agent1_server.service_definition
)

# agent2 = OpenAIAgent.from_tools(
#     [agent1_server_tool],
#     system_prompt="Perform the task, return the result as well as a funny joke.",
# )  # worker2.as_agent()

agent2 = ReActAgent.from_tools([agent1_server_tool],
                               llm=llm_gpt4o, verbose=True)

agent2_server = AgentService(
    agent=agent2,
    message_queue=message_queue,
    description="Useful for telling funny jokes.",
    service_name="dumb_fact_agent",
)

print(agent1_server.service_definition)
print(agent2_server.service_definition)
agent2_component = ServiceComponent.from_service_definition(agent2_server.service_definition)

pipeline = QueryPipeline(chain=[agent2_component])

pipeline_orchestrator = PipelineOrchestrator(pipeline)

control_plane = ControlPlaneServer(message_queue, pipeline_orchestrator)

# launch it
launcher = LocalLauncher([agent1_server, agent2_server], control_plane, message_queue)
result = launcher.launch_single("What is the secret fact?")

print(f"Result: {result}")
