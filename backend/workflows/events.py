from pathlib import Path
from typing import List

from llama_index.core.workflow import Event

from workflows.paper_scraping import Paper, IsCitationRelevant
from workflows.schemas import *


class TavilyResultsEvent(Event):
    results: List[TavilySearchResult]


class PaperEvent(Event):
    paper: Paper


class FilteredPaperEvent(Event):
    paper: Paper
    is_relevant: IsCitationRelevant


class FilteredPapersEvent(Event):
    paper: Paper


class Paper2SummaryDispatcherEvent(Event):
    papers_path: str


class Paper2SummaryEvent(Event):
    pdf_path: Path
    image_output_dir: Path
    summary_path: Path


class SummaryStoredEvent(Event):
    fpath: Path


class SummaryWfReadyEvent(Event):
    summary_dir: str


class SummaryEvent(Event):
    summary: str


class OutlineFeedbackEvent(Event):
    summary: str
    outline: SlideOutline
    feedback: str


class OutlineEvent(Event):
    summary: str
    outline: SlideOutline


class OutlineOkEvent(Event):
    summary: str
    outline: SlideOutline


class OutlinesWithLayoutEvent(Event):
    # outline_w_layout: List[SlideOutlineWithLayout]
    outlines_fpath: Path
    outline_example: SlideOutlineWithLayout


class ConsolidatedOutlineEvent(Event):
    outlines: List[SlideOutline]


class PythonCodeEvent(Event):
    code: str


class SlideGeneratedEvent(Event):
    pptx_fpath: str  # remote pptx path


class SlideValidationEvent(Event):
    result: SlideValidationResult


class DummyEvent(Event):
    result: str
