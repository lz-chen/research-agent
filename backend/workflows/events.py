from pathlib import Path
from typing import List, Optional, Any

from llama_index.core.workflow import Event
from pydantic import BaseModel, Field

from workflows.paper_scraping import Paper, IsCitationRelevant


class SlideOutline(BaseModel):
    """Slide outline for one page"""

    title: str = Field(..., description="Title of the slide")
    content: str = Field(..., description="Main text content of the slide")


class SlideOutlineWithLayout(BaseModel):
    """Slide outline with layout information for one page"""

    title: str = Field(..., description="Title of the slide")
    content: str = Field(..., description="Main text content of the slide")
    layout_name: str = Field(
        ..., description="Name of the page layout to be used for the slide"
    )
    idx_title_placeholder: str = Field(
        ..., description="Index of the title placeholder in the page layout"
    )
    idx_content_placeholder: str = Field(
        ..., description="Index of the content placeholder in the page layout"
    )


class SlideValidationResult(BaseModel):
    is_valid: bool
    suggestion_to_fix: str


class TavilySearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[Any]


class TavilyResultsEvent(Event):
    results: List[TavilySearchResult]


class PaperEvent(Event):
    paper: Paper


class FilteredPaperEvent(Event):
    paper: Paper
    is_relevant: IsCitationRelevant


class FilteredPapersEvent(Event):
    paper: Paper


class DownloadPaperEvent(Event):
    papers_dict: dict


class Paper2SummaryEvent(Event):
    papers_path: str


class SummaryStoredEvent(Event):
    fapth: str


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
