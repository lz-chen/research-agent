import re
from pathlib import Path
from typing import Optional

import click
import qdrant_client
from llama_index.core.agent import ReActAgent, ReActChatFormatter, FunctionCallingAgentWorker
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms import LLM
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.node_parser import (
    UnstructuredElementNodeParser,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.workflow import Workflow, step
from llama_index.readers.file import FlatReader, PDFReader
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core import Settings
from config import settings
from prompts import LLAMAPARSE_INSTRUCTION, SUMMARIZE_PAPER_PMT_REACT
from services.llms import llm_gpt4o, new_gpt4o
from services.embeddings import aoai_embedder
import logging
import sys
from llama_index.core import PromptTemplate
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step, Event

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

Settings.llm = llm_gpt4o
Settings.embed_model = aoai_embedder


def read_summary_content(file_path: Path):
    """
    Read the content of the summary file
    :param file_path: Path to the summary file
    :return: Content of the summary file
    """
    with file_path.open("r") as file:
        return file.read()


class SummaryEvent(Event):
    summary: str


class OutlineEvent(Event):
    outline: str


class ConsolidatedOutlineEvent(Event):
    outline: str


class SlideGenWorkflow(Workflow):

    @step(pass_context=True)
    def get_summaries(self, ctx: Context, ev: StartEvent) -> SummaryEvent:
        file_path = ctx.data["file_dir"]
        markdown_files = list(file_path.glob("*.md"))
        ctx.data["n_summaries"] = len(markdown_files)
        for f in markdown_files:
            s = read_summary_content(f)
            self.send_event(SummaryEvent(summary=s))

    @step(pass_context=True)
    def summary2outline(self, ctx: Context, ev: SummaryEvent) -> OutlineEvent:
        system_prompt = """"
            You are an AI specialized in generating PowerPoint slide outlines based on the content provided.
            You will receive a markdown string that contains the summary of papers and 
            you will generate a slide outlines for each paper.
            Requirements:
            - One slide page per paper summary
            - Use the paper title as the slide title
            - Use the summary in the markdown file as the slide content, 
              keep the key headers as bullet points but you can 
              rephrase the content to make it more consise and readable
        """
        # agent_worker = FunctionCallingAgentWorker.from_tools(
        #     tools=[],
        #     llm=llm_gpt4o,
        #     allow_parallel_tool_calls=False,
        #     system_prompt=system_prompt
        # )
        # agent = agent_worker.as_agent()
        # response = agent.chat(ev.summary)

        llm = new_gpt4o(0.1)
        llm.system_prompt = system_prompt
        response = llm.chat(ev.summary)
        return OutlineEvent(outline=response.text)

    @step(pass_context=True)
    def consolidate_outlines(self, ctx: Context, ev: OutlineEvent) -> ConsolidatedOutlineEvent:
        ready = ctx.collect_events(ev, [OutlineEvent] * ctx.data["n_summaries"])
        if ready is None:
            return None
        system_prompt = """"
            You are an AI that consolidate pieces of slide content to a full slide outline.
            You will receive a list of content items. Merge them to one outline with following requirements:
            - One slide page per item
            - Keep the individual content as is
        """

    @step(pass_context=True)
    def outline2code(self, ctx):
        pass

    @step(pass_context=True)
    def execute_code(self, ctx):
        pass


@click.command()
@click.option("--file_dir", "-d", required=True,
              help="Path to the directory that contains paper summaries for generating slide outlines",
              default="./data/summaries")
def main(file_dir: str):
    create_agent(file_dir)


if __name__ == "__main__":
    main()
