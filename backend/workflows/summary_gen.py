import asyncio
import json
import random
import string
from pathlib import Path

import click
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import FunctionCallingProgram, MultiModalLLMCompletionProgram
from llama_index.core.tools import FunctionTool
from llama_index.tools.tavily_research import TavilyToolSpec
from tavily import TavilyClient

from config import settings
from prompts.prompts import SLIDE_GEN_PMT, REACT_PROMPT_SUFFIX, SUMMARY2OUTLINE_PMT, AUGMENT_LAYOUT_PMT, \
    SLIDE_VALIDATION_PMT, \
    SLIDE_MODIFICATION_PMT, MODIFY_SUMMARY2OUTLINE_PMT
from services.llms import llm_gpt4o, new_gpt4o, new_gpt4o_mini, mm_gpt4o
from services.embeddings import aoai_embedder
import logging
import sys
from llama_index.core import PromptTemplate
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step, draw_all_possible_flows

from utils.tools import get_all_layouts_info
import inspect
from llama_index.tools.azure_code_interpreter import (
    AzureCodeInterpreterToolSpec,
)

from utils.file_processing import pptx2images
from workflows.events import *
from workflows.paper_scraping import get_paper_with_citations, process_citation, download_relevant_citations

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

Settings.llm = llm_gpt4o
Settings.embed_model = aoai_embedder


class SummaryGenerationWorkflow(Workflow):
    tavily_max_results: int = 2
    n_max_final_papers: int = 10
    papers_download_path: str = "./data/papers"

    @step
    def tavily_query(self, ev: StartEvent) -> TavilyResultsEvent:
        query = f"arxiv papers about the state of the art of {ev.user_query}"
        # query = f"Help me find the papers on arxiv on the topic of recent advance on {ev.user_query}"
        tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        response = tavily_client.search(query, max_results=self.tavily_max_results)
        results = [TavilySearchResult(**d) for d in response["results"]]
        return TavilyResultsEvent(results=results)

    @step(pass_context=True)
    def get_paper_with_citations(self, ctx: Context, ev: TavilyResultsEvent) -> PaperEvent:
        papers = []
        for r in ev.results:
            p = get_paper_with_citations(r.title)
            papers += p
        # deduplicate papers
        papers = list({p.entry_id: p for p in papers}.values())
        ctx.data["n_all_papers"] = len(papers)
        for paper in papers:
            self.send_event(PaperEvent(paper=paper))

    @step()
    async def filter_papers(self, ev: PaperEvent) -> FilteredPaperEvent:
        llm = new_gpt4o_mini(temperature=0.0)
        _, response = await process_citation(0, ev.paper, llm)
        return FilteredPaperEvent(paper=ev.paper, is_relevant=response)

    @step(pass_context=True)
    def download_papers(self, ctx: Context, ev: FilteredPaperEvent) -> DownloadPaperEvent:
        ready = ctx.collect_events(ev, [FilteredPaperEvent] * ctx.data["n_all_papers"])
        if ready is None:
            return None
        papers = sorted(ready, key=lambda x: x.is_relevant.score, reverse=True)[:self.n_max_final_papers]
        papers_dict = {i: {"citation": p.paper, "is_relevant": p.is_relevant} for i, p in enumerate(papers)}
        return DownloadPaperEvent(papers_dict=papers_dict)

    @step()
    def download_paper(self, ev: DownloadPaperEvent) -> StopEvent:
        download_relevant_citations(ev.papers_dict, Path(self.papers_download_path))
        return StopEvent()


async def run_workflow(user_query: str):
    wf = SummaryGenerationWorkflow(
        timeout=1200, verbose=True)
    result = await wf.run(
        user_query=user_query,
    )
    print(result)


@click.command()
@click.option("--user-query", "-q", required=False,
              help="The user query",
              default="powerpoint slides automation")
def main(user_query: str):
    asyncio.run(run_workflow(user_query))


if __name__ == "__main__":
    draw_all_possible_flows(SummaryGenerationWorkflow, filename="summary_gen_flows.html")
    main()
