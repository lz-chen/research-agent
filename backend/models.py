# models.py
from pydantic import BaseModel, Field


class SlideGenFileDirectory(BaseModel):
    path: str = Field(..., example="path/to/file")
