import asyncio
import inspect
import random
import string

import click
import mlflow
from llama_index.core import Settings
from tavily import TavilyClient

from config import settings
from services.llms import llm_gpt4o, new_gpt4o_mini
from services.embeddings import aoai_embedder
import logging
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    draw_all_possible_flows,
)

from utils.file_processing import pdf2images
from workflows.events import *
from workflows.paper_scraping import (
    get_paper_with_citations,
    process_citation,
    download_relevant_citations,
)
from workflows.summary_using_images import (
    get_summary_from_gpt4o,
    save_summary_as_markdown,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


Settings.llm = llm_gpt4o
Settings.embed_model = aoai_embedder


class SummaryGenerationWorkflow(Workflow):
    tavily_max_results: int = 2
    n_max_final_papers: int = 10

    def __init__(self, *args, **kwargs):
        # self.parent_workflow = None

        super().__init__(*args, **kwargs)
        # make random string of length 10 and make it a suffix for WORKFLOW_ARTIFACTS_PATH
        class_name = self.__class__.__name__
        s = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self.workflow_artifacts_path = (
            Path(settings.WORKFLOW_ARTIFACTS_PATH).joinpath(class_name).joinpath(s)
        )
        self.papers_download_path = self.workflow_artifacts_path.joinpath(
            settings.PAPERS_DOWNLOAD_PATH
        )
        self.papers_download_path.mkdir(parents=True, exist_ok=True)
        self.papers_images_path = self.workflow_artifacts_path.joinpath(
            settings.PAPERS_IMAGES_PATH
        )
        self.papers_images_path.mkdir(parents=True, exist_ok=True)
        self.paper_summary_path = self.workflow_artifacts_path.joinpath(
            settings.PAPERS_IMAGES_PATH
        )
        self.paper_summary_path.mkdir(parents=True, exist_ok=True)

    async def run(self, *args, **kwargs):
        self.loop = asyncio.get_running_loop()  # Store the event loop
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment("SummaryGenerationWorkflow")
        mlflow.llama_index.autolog()
        mlflow.start_run()
        result = await super().run(*args, **kwargs)
        mlflow.end_run()
        return result

    @step(pass_context=True)
    async def tavily_query(self, ctx: Context, ev: StartEvent) -> TavilyResultsEvent:
        query = f"arxiv papers about the state of the art of {ev.user_query}"
        ctx.write_event_to_stream(
            Event(
                msg=WorkflowStreamingEvent(
                    event_type="server_message",
                    event_sender=inspect.currentframe().f_code.co_name,
                    event_content={"message": f"Querying Tavily with: '{query}'"},
                ).json()
            )
        )
        tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        response = tavily_client.search(query, max_results=self.tavily_max_results)
        results = [TavilySearchResult(**d) for d in response["results"]]
        logger.info(f"tavily results: {results}")
        return TavilyResultsEvent(results=results)

    @step(pass_context=True)
    async def get_paper_with_citations(
        self, ctx: Context, ev: TavilyResultsEvent
    ) -> PaperEvent:
        papers = []
        for r in ev.results:
            p = get_paper_with_citations(r.title)
            papers += p
        # deduplicate papers
        papers = list({p.entry_id: p for p in papers}.values())
        ctx.data["n_all_papers"] = len(papers)
        logger.info(f"papers found on ss: {[p.title for p in papers]}")
        for paper in papers:
            ctx.write_event_to_stream(
                Event(
                    msg=WorkflowStreamingEvent(
                        event_type="server_message",
                        event_sender=inspect.currentframe().f_code.co_name,
                        event_content={
                            "message": f"Found related paper: {paper.title}"
                        },
                    ).json()
                )
            )
            self.send_event(PaperEvent(paper=paper))

    @step(num_workers=4)
    async def filter_papers(self, ev: PaperEvent) -> FilteredPaperEvent:
        llm = new_gpt4o_mini(temperature=0.0)
        _, response = await process_citation(0, ev.paper, llm)
        return FilteredPaperEvent(paper=ev.paper, is_relevant=response)

    @step(pass_context=True)
    async def download_papers(
        self, ctx: Context, ev: FilteredPaperEvent
    ) -> Paper2SummaryDispatcherEvent:
        ready = ctx.collect_events(ev, [FilteredPaperEvent] * ctx.data["n_all_papers"])
        if ready is None:
            return None
        papers = sorted(
            ready,
            key=lambda x: (
                x.is_relevant.score,  # prioritize papers with higher score
                "ArXiv"
                in (
                    x.paper.external_ids or {}
                ),  # prioritize papers can be found on ArXiv
            ),
            reverse=True,
        )[: self.n_max_final_papers]
        papers_dict = {
            i: {"citation": p.paper, "is_relevant": p.is_relevant}
            for i, p in enumerate(papers)
        }
        ctx.write_event_to_stream(
            Event(
                msg=WorkflowStreamingEvent(
                    event_type="server_message",
                    event_sender=inspect.currentframe().f_code.co_name,
                    event_content={
                        "message": f"Downloading filtered relevant papers:\n"
                        f"{' | '.join([p.paper.title for p in papers])}"
                    },
                ).json()
            )
        )
        download_relevant_citations(papers_dict, Path(self.papers_download_path))
        return Paper2SummaryDispatcherEvent(
            papers_path=self.papers_download_path.as_posix()
        )

    @step(pass_context=True)
    async def paper2summary_dispatcher(
        self, ctx: Context, ev: Paper2SummaryDispatcherEvent
    ) -> Paper2SummaryEvent:
        ctx.data["n_pdfs"] = 0
        for pdf_name in Path(ev.papers_path).glob("*.pdf"):
            img_output_dir = self.papers_images_path / pdf_name.stem
            img_output_dir.mkdir(exist_ok=True, parents=True)
            summary_fpath = self.paper_summary_path / f"{pdf_name.stem}.md"
            ctx.data["n_pdfs"] += 1
            self.send_event(
                Paper2SummaryEvent(
                    pdf_path=pdf_name,
                    image_output_dir=img_output_dir,
                    summary_path=summary_fpath,
                )
            )

    @step(num_workers=4, pass_context=True)
    async def paper2summary(
        self, ctx: Context, ev: Paper2SummaryEvent
    ) -> SummaryStoredEvent:
        pdf2images(ev.pdf_path, ev.image_output_dir)
        summary_txt = await get_summary_from_gpt4o(ev.image_output_dir)
        save_summary_as_markdown(summary_txt, ev.summary_path)
        ctx.write_event_to_stream(
            Event(
                msg=WorkflowStreamingEvent(
                    event_type="server_message",
                    event_sender=inspect.currentframe().f_code.co_name,
                    event_content={"message": f"Summarizing paper: {ev.pdf_path}"},
                ).json()
            )
        )
        return SummaryStoredEvent(fpath=ev.summary_path)

    @step(pass_context=True)
    async def finish(self, ctx: Context, ev: SummaryStoredEvent) -> StopEvent:
        ready = ctx.collect_events(ev, [SummaryStoredEvent] * ctx.data["n_pdfs"])
        if ready is None:
            return None
        for e in ready:
            assert e.fpath.is_file()
        logger.info(f"All summary are stored!")
        return StopEvent(result=e.fpath.parent.as_posix())


# workflow for debugging purpose
class SummaryGenerationDummyWorkflow(Workflow):
    @step
    async def dummy_start_step(self, ev: StartEvent) -> DummyEvent:
        return DummyEvent(result="dummy")

    @step
    async def dummy_stop_step(self, ev: DummyEvent) -> StopEvent:
        return StopEvent(
            result="workflow_artifacts/SummaryGenerationWorkflow/5sn92wndsx/data/paper_summaries"
        )


async def run_workflow(user_query: str):
    wf = SummaryGenerationWorkflow(timeout=1200, verbose=True)
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
        SummaryGenerationWorkflow, filename="summary_gen_flows.html"
    )
    main()
