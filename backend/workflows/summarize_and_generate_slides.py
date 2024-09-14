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

from workflows.slide_gen import SlideGenerationWorkflow
from workflows.summary_gen import SummaryGenerationWorkflow

Settings.llm = llm_gpt4o
Settings.embed_model = aoai_embedder


class SummaryAndSlideGenerationWorkflow(Workflow):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize user input future
        self.user_input_future = asyncio.Future()

    async def run(self, *args, **kwargs):
        self.loop = asyncio.get_running_loop()  # Store the event loop
        return await super().run(*args, **kwargs)

    async def reset_user_input_future(self):
        self.user_input_future = self.loop.create_future()

    async def run_subworkflow(self, sub_wf, ctx, **kwargs):
        sub_wf.user_input_future = self.user_input_future
        sub_wf.parent_workflow = self
        # Start the sub-workflow
        sub_task = asyncio.create_task(sub_wf.run(**kwargs))
        # While the sub-workflow is running, relay its events
        async for event in sub_wf.stream_events():
            ctx.write_event_to_stream(event)
        # Wait for the sub-workflow to complete
        result = await sub_task
        return result

    @step
    async def summary_gen(
            self, ctx: Context, ev: StartEvent, summary_gen_wf: SummaryGenerationWorkflow
    ) -> SummaryWfReadyEvent:
        # res = await summary_gen_wf.run(user_query=ev.user_query)
        res = await self.run_subworkflow(summary_gen_wf, ctx, user_query=ev.user_query)

        return SummaryWfReadyEvent(summary_dir=res)

    @step
    async def slide_gen(
            self, ctx: Context, ev: SummaryWfReadyEvent, slide_gen_wf: SlideGenerationWorkflow
    ) -> StopEvent:
        # res = await slide_gen_wf.run(file_dir=ev.summary_dir)
        res = await self.run_subworkflow(slide_gen_wf, ctx, file_dir=ev.summary_dir)

        return StopEvent(res)


async def run_workflow(user_query: str):
    wf = SummaryAndSlideGenerationWorkflow(timeout=2000, verbose=True)
    wf.add_workflows(
        summary_gen_wf=SummaryGenerationWorkflow(timeout=800, verbose=True)
    )
    wf.add_workflows(slide_gen_wf=SlideGenerationWorkflow(timeout=1200, verbose=True))
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
