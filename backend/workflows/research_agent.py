import asyncio
import json
import random
import string
from pathlib import Path

import click
from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import FunctionCallingProgram, MultiModalLLMCompletionProgram
from llama_index.core.tools import FunctionTool

from config import settings
from prompts.prompts import SLIDE_GEN_PMT, REACT_PROMPT_SUFFIX, SUMMARY2OUTLINE_PMT, AUGMENT_LAYOUT_PMT, \
    SLIDE_VALIDATION_PMT, \
    SLIDE_MODIFICATION_PMT, MODIFY_SUMMARY2OUTLINE_PMT
from services.llms import llm_gpt4o, new_gpt4o, new_gpt4o_mini, mm_gpt4o
from services.embeddings import aoai_embedder
import logging
import sys
from llama_index.core import PromptTemplate
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step, draw_all_possible_flows

from utils.tools import get_all_layouts_info
import inspect
from llama_index.tools.azure_code_interpreter import (
    AzureCodeInterpreterToolSpec,
)

from utils.file_processing import pptx2images
from workflows.events import *

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

Settings.llm = llm_gpt4o
Settings.embed_model = aoai_embedder


class ResearchAgentWorkflow(Workflow):
    summary_gen_workflow = None
    slide_gen_workflow = None
