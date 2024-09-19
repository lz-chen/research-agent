from typing import Optional, Any, Literal, Dict

from pydantic import BaseModel, Field


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


class SlideNeedModifyResult(BaseModel):
    slide_idx: int
    suggestion_to_fix: str


class TavilySearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[Any]


class WorkflowStreamingEvent(BaseModel):
    event_type: Literal["server_message", "request_user_input"] = Field(
        ..., description="Type of the event"
    )
    event_sender: str = Field(
        ..., description="Sender (workflow step name) of the event"
    )
    event_content: Dict[str, Any] = Field(..., description="Content of the event")
