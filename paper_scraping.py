import asyncio
from pathlib import Path

import click
from llama_index.core.program import FunctionCallingProgram

from config import settings

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import arxiv
from semanticscholar import SemanticScholar
from prompts import IS_CITATION_RELEVANT_PMT

from services import llms
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Paper(BaseModel):
    entry_id: str
    title: str
    authors: List[str]
    published: str
    summary: str
    primary_category: Optional[str] = None
    link: Optional[str] = None
    external_ids: Optional[dict] = None
    open_access_pdf: Optional[dict] = None


class IsCitationRelevant(BaseModel):
    # is_relevant: bool
    score: int
    reason: str


def search_paper_arxiv(title, limit=1):
    client = arxiv.Client()

    search = arxiv.Search(
        query=title,
        max_results=limit,
        # sort_by=arxiv.SortCriterion.SubmittedDate
    )

    results = client.results(search)
    papers = []
    for result in results:
        paper = Paper(
            entry_id=result.entry_id,
            title=result.title,
            authors=[a.name for a in result.authors],
            published=result.published.strftime("%Y-%m-%d %H:%M:%S"),
            summary=result.summary,
            primary_category=result.primary_category,
            link=result.pdf_url
        )
        papers.append(paper)
    return papers


def search_paper_ss(query: str, limit: int = 1):
    s2 = SemanticScholar()
    results = s2.search_paper(query, limit=limit)
    papers = []
    for result in results.raw_data:
        paper = Paper(
            entry_id=str(result['corpusId']),
            title=result['title'],
            authors=[a["name"] for a in result['authors']],
            published=result['publicationDate'],
            summary=result['abstract'],
            primary_category=result['fieldsOfStudy'][0],
            link=result['url'],
            external_ids=result['externalIds'],
            open_access_pdf=result['openAccessPdf']
        )
        papers.append(paper)

    return papers


def get_citations_ss(paper: Paper):
    s2 = SemanticScholar(api_key=settings.SEMANTIC_SCHOLAR_API_KEY)
    results = s2.get_paper_citations(f"CorpusID:{paper.entry_id}")
    citations = []
    for res in results:
        result = res.paper
        try:
            citation = Paper(
                entry_id=str(result['corpusId']),
                title=result['title'],
                authors=[a["name"] for a in result['authors']],
                published=result['publicationDate'].strftime("%Y-%m-%d %H:%M:%S") if not isinstance(
                    result['publicationDate'], str) else result['publicationDate'],
                summary=result['abstract'],
                primary_category=result['fieldsOfStudy'][0] if result['fieldsOfStudy'] else None,
                link=result['url'],
                external_ids=result['externalIds'],
                open_access_pdf=result['openAccessPdf']
            )
            citations.append(citation)
        except Exception as e:
            logging.error("Error parsing citation.")
            logging.error(e)
            continue
    return citations


def get_paper_with_citations(query: str, limit: int = 1) -> List[Paper]:
    """
    Get a paper and its citations from Semantic Scholar.
    :param query: title of the paper to query
    :param limit: maximum number of papers to return for initial query. Default is 1.
    :return:
    """
    papers = search_paper_ss(query, limit=limit)
    logging.info("Found paper!")
    logging.debug(papers[0].dict())
    citations = get_citations_ss(papers[0])
    logging.info(f"Found {len(citations)} citations!")
    logging.debug([c.dict() for c in citations])
    citations.append(papers[0])
    return citations


async def process_citation(i, citation, llm):
    program = FunctionCallingProgram.from_defaults(
        llm=llm,
        output_cls=IsCitationRelevant,
        prompt_template_str=IS_CITATION_RELEVANT_PMT,
        verbose=True,
    )
    response = await program.acall(
        title=citation.title,
        abstract=citation.summary,
        description="Data model for whether the paper is relevant to the research topic.",
    )
    return i, response


async def filter_relevant_citations(citations: List[Paper]) -> Dict[str, Any]:
    """
    Filter relevant citations based on research topic.
    :param citations: list of papers to filter
    :param research_topic: research topic to filter by
    :return:
    """
    llm = llms.new_gpt4o(temperature=0.0)
    tasks = [process_citation(i, citation, llm) for i, citation in enumerate(citations)]
    results = await asyncio.gather(*tasks)

    # Match results to their corresponding index
    citations_w_relevance = {}
    for i, response in results:
        citations_w_relevance[i] = {
            "citation": citations[i],
            "is_relevant": response,
        }

    return citations_w_relevance


def download_paper_arxiv(paper_id: str, download_dir: str, filename: str):
    paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id])))
    logging.info(f"Downloading paper: {paper.title} to {download_dir}/{filename}")
    paper.download_pdf(dirpath=download_dir, filename=filename)
    logging.info("Done!")


def download_relevant_citations(citation_dict: Dict[str, Any]):
    """
    Check of the citation is relevant and download the relevant ones use arxiv.
    :param citation_dict:
    :return:
    """
    paper_dir = Path(__file__).parent / "data" / "papers"
    paper_dir.mkdir(parents=True, exist_ok=True)
    # count how many relevant citations are there
    relevant_citations = len([v["is_relevant"].score for v in citation_dict.values() if v["is_relevant"].score > 0])
    logging.info(f"Found {relevant_citations} relevant citations.")

    for i, v in citation_dict.items():
        if v["is_relevant"].score > 0:
            if v["citation"].external_ids and "ArXiv" in v["citation"].external_ids:
                arxiv_id = v["citation"].external_ids["ArXiv"]
                download_paper_arxiv(arxiv_id,
                                     paper_dir.as_posix(),
                                     f"{v['citation'].title}.pdf")
    return paper_dir


def paper2md(fname: Path, output_dir: Path, langs: Optional[list] = ["English"], max_pages: Optional[int] = None,
             batch_multiplier: Optional[int] = 1, start_page: Optional[int] = None):
    # https://github.com/VikParuchuri/marker/blob/master/convert_single.py#L8
    from marker.convert import convert_single_pdf
    from marker.models import load_all_models
    from marker.output import save_markdown
    import time

    model_lst = load_all_models()
    start = time.time()
    full_text, images, out_meta = convert_single_pdf(fname.as_posix(), model_lst, max_pages=max_pages, langs=langs,
                                                     batch_multiplier=batch_multiplier, start_page=start_page)

    name = fname.name
    subfolder_path = save_markdown(output_dir.as_posix(), name, full_text, images, out_meta)

    logging.info(f"Saved markdown to the '{subfolder_path}' folder")
    logging.debug(f"Total time: {time.time() - start}")
    # return subfolder_path


def parse_pdf(pdf_path: Path, force_reparse=False):
    md_output_dir = pdf_path.parents[1].joinpath("parsed_papers")

    if len(list(md_output_dir.joinpath(pdf_path.stem).glob("*.md"))) and not force_reparse:
        logging.info(f"force_reparse=False and Markdown file already exists for '{pdf_path}', skipping...")
    else:
        logging.info(f"Converting '{pdf_path}' to markdown")
        paper2md(Path(pdf_path), md_output_dir)
    return md_output_dir


def parse_paper_pdfs(papers_dir: Path, force_reparse=False):
    for f in papers_dir.rglob("*.pdf"):
        if f.parents[1].joinpath("summaries").joinpath(f"{f.stem}_summary.md").exists():
            logging.info(f"Summary already exists for '{f}', skip generation")
            continue
        logging.info(f"Parsing pdf file '{f}' to markdown")
        parse_pdf(f, force_reparse)




@click.command()
@click.argument('entry_paper_title', type=str,
                default="DOC2PPT: Automatic Presentation Slides Generation from Scientific Documents")
def main(entry_paper_title):
    citations = get_paper_with_citations(entry_paper_title)
    if citations:
        relevant_citations = asyncio.run(filter_relevant_citations(citations))
        paper_dir = download_relevant_citations(relevant_citations)
        parse_paper_pdfs(paper_dir)
    # pprint(relevant_citations)


if __name__ == '__main__':
    main()
