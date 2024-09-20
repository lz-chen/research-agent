# models.py
from pydantic import BaseModel, Field


class SlideGenFileDirectory(BaseModel):
    path: str = Field(..., example="path/to/file")


class ResearchTopic(BaseModel):
    query: str = Field(..., example="example query")
