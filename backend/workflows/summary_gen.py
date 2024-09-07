import asyncio
import random
import string

import click
from llama_index.core import Settings
from tavily import TavilyClient

from config import settings
from services.llms import llm_gpt4o, new_gpt4o_mini
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # make random string of length 10 and make it a suffix for WORKFLOW_ARTIFACTS_PATH
        class_name = self.__class__.__name__
        s = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self.workflow_artifacts_path = (
            Path(settings.WORKFLOW_ARTIFACTS_PATH).joinpath(class_name).joinpath(s)
        )
        self.papers_download_path = self.workflow_artifacts_path.joinpath("data/papers")
        self.papers_download_path.mkdir(parents=True, exist_ok=True)
        self.papers_images_path = self.workflow_artifacts_path.joinpath(
            "data/papers_images"
        )
        self.papers_images_path.mkdir(parents=True, exist_ok=True)
        self.paper_summary_path = self.workflow_artifacts_path.joinpath(
            "data/paper_summaries"
        )
        self.paper_summary_path.mkdir(parents=True, exist_ok=True)

    @step
    def tavily_query(self, ev: StartEvent) -> TavilyResultsEvent:
        query = f"arxiv papers about the state of the art of {ev.user_query}"
        # query = f"Help me find the papers on arxiv on the topic of recent advance on {ev.user_query}"
        tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
        response = tavily_client.search(query, max_results=self.tavily_max_results)
        results = [TavilySearchResult(**d) for d in response["results"]]
        return TavilyResultsEvent(results=results)

    @step(pass_context=True)
    def get_paper_with_citations(
        self, ctx: Context, ev: TavilyResultsEvent
    ) -> PaperEvent:
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
    def download_papers(
        self, ctx: Context, ev: FilteredPaperEvent
    ) -> Paper2SummaryDispatcherEvent:
        ready = ctx.collect_events(ev, [FilteredPaperEvent] * ctx.data["n_all_papers"])
        if ready is None:
            return None
        papers = sorted(
            ready,
            key=lambda x: (
                x.is_relevant.score,  # prioritize papers with higer score
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
        download_relevant_citations(papers_dict, Path(self.papers_download_path))
        return Paper2SummaryDispatcherEvent(
            papers_path=self.papers_download_path.as_posix()
        )

    @step(pass_context=True)
    def paper2summary_dispatcher(
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

    @step()
    async def paper2summary(self, ev: Paper2SummaryEvent) -> SummaryStoredEvent:
        pdf2images(ev.pdf_path, ev.image_output_dir)
        summary_txt = get_summary_from_gpt4o(ev.image_output_dir)
        save_summary_as_markdown(summary_txt, ev.summary_path)
        return SummaryStoredEvent(fapth=ev.summary_path)

    @step(pass_context=True)
    def finish(self, ctx: Context, ev: SummaryStoredEvent) -> StopEvent:
        ready = ctx.collect_events(ev, [SummaryStoredEvent] * ctx.data["n_pdfs"])
        if ready is None:
            return None
        for e in ready:
            assert e.fpath.is_file()
        logging.info(f"All summary are stored!")
        return StopEvent()


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
