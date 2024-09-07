import asyncio

import click
from llama_index.core import Settings

from services.llms import llm_gpt4o
from services.embeddings import aoai_embedder
import logging
import sys
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    draw_all_possible_flows,
)

from workflows.events import *

from workflows.slide_gen import SlideGenWorkflow
from workflows.summary_gen import SummaryGenerationWorkflow


logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

Settings.llm = llm_gpt4o
Settings.embed_model = aoai_embedder


class SummaryAndSlideGenerationWorkflow(Workflow):
    @step
    async def summary_gen(
        self, ctx: Context, ev: StartEvent, summary_gen_wf: SummaryGenerationWorkflow
    ) -> SummaryWfReadyEvent:
        print("Need to run reflection")
        res = await summary_gen_wf.run(user_query=ev.user_query)
        return SummaryWfReadyEvent(summary_dir=res)

    @step
    async def slide_gen(
        self, ctx: Context, ev: SummaryWfReadyEvent, slide_gen_wf: SlideGenWorkflow
    ) -> StopEvent:
        res = await slide_gen_wf.run(file_dir=ev.summary_dir)
        return StopEvent()


async def run_workflow(user_query: str):
    wf = SummaryAndSlideGenerationWorkflow(timeout=2000, verbose=True)
    wf.add_workflows(
        summary_gen_wf=SummaryGenerationWorkflow(timeout=800, verbose=True)
    )
    wf.add_workflows(slide_gen_wf=SlideGenWorkflow(timeout=1200, verbose=True))
    result = await wf.run(
        user_query=user_query,
    )
    print(result)


@click.command()
@click.option(
    "--user-query",
    "-q",
    required=False,
    help="The user query",
    default="powerpoint slides automation",
)
def main(user_query: str):
    asyncio.run(run_workflow(user_query))


if __name__ == "__main__":
    draw_all_possible_flows(
        SummaryAndSlideGenerationWorkflow, filename="summary_slide_gen_flows.html"
    )
    main()
