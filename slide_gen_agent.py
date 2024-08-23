import asyncio
import re
from pathlib import Path
from typing import Optional, List

import click
# import qdrant_client
# from llama_index.core.agent import ReActAgent, ReActChatFormatter, FunctionCallingAgentWorker
# from llama_index.core.ingestion import IngestionPipeline
# from llama_index.core.llms import LLM
# from llama_index.core.node_parser import MarkdownElementNodeParser
# from llama_index.core.node_parser import (
#     UnstructuredElementNodeParser,
# )
# from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
# from llama_index.core.workflow import Workflow, step
# from llama_index.readers.file import FlatReader, PDFReader
# from llama_index.storage.docstore.redis import RedisDocumentStore
# from llama_index.storage.index_store.redis import RedisIndexStore
# from llama_index.vector_stores.qdrant import QdrantVectorStore
# from llama_parse import LlamaParse
# from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent, FunctionCallingAgentWorker
from llama_index.core.program import FunctionCallingProgram
from llama_index.core.tools import FunctionTool

from config import settings
from prompts import SLIDE_GEN_PMT, REACT_PROMPT_SUFFIX, SUMMARY2OUTLINE_PMT, AUGMENT_LAYOUT_PMT, LAYOUT2CODE_PMT
from services.llms import llm_gpt4o, new_gpt4o
from services.embeddings import aoai_embedder
import logging
import sys
from llama_index.core import PromptTemplate
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step, Event, draw_all_possible_flows
from pydantic import BaseModel, Field

from tools import get_all_layouts_info

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

Settings.llm = llm_gpt4o
Settings.embed_model = aoai_embedder


def read_summary_content(file_path: Path):
    """
    Read the content of the summary file
    :param file_path: Path to the summary file
    :return: Content of the summary file
    """
    with file_path.open("r") as file:
        return file.read()


class SlideOutline(BaseModel):
    """Slide outline for one page"""
    title: str = Field(..., description="Title of the slide")
    content: str = Field(..., description="Main text content of the slide")


class SlideOutlineWithLayout(BaseModel):
    """Slide outline with layout information for one page"""
    title: str = Field(..., description="Title of the slide")
    content: str = Field(..., description="Main text content of the slide")
    layout_name: str = Field(..., description="Name of the page layout to be used for the slide")
    idx_title_placeholder: str = Field(..., description="Index of the title placeholder in the page layout")
    idx_content_placeholder: str = Field(..., description="Index of the content placeholder in the page layout")


class SummaryEvent(Event):
    summary: str


class OutlineEvent(Event):
    outline: SlideOutline


class OutlinesWithLayoutEvent(Event):
    outline_w_layout: List[SlideOutlineWithLayout]


class ConsolidatedOutlineEvent(Event):
    outlines: List[SlideOutline]


class PythonCodeEvent(Event):
    code: str


class SlideGenWorkflow(Workflow):

    @step(pass_context=True, num_workers=4)
    def get_summaries(self, ctx: Context, ev: StartEvent) -> SummaryEvent:
        """Entry point of the workflow. Read the content of the summary files from provided
        directory. For each summary file, send a SummaryEvent to the next step."""

        ctx.data["ppt_template_path"] = ev.get("ppt_template_path")
        ctx.data["final_pptx_fname"] = ev.get("final_pptx_fname")

        markdown_files = list(Path(ev.get("file_dir")).glob("*.md"))
        ctx.data["n_summaries"] = len(markdown_files)
        for f in markdown_files:
            s = read_summary_content(f)
            self.send_event(SummaryEvent(summary=s))

    @step(pass_context=True)
    async def summary2outline(self, ctx: Context, ev: SummaryEvent) -> OutlineEvent:
        """Convert the summary content of one paper to slide outline of one page."""
        # prompt = """"
        #     You are an AI specialized in generating PowerPoint slide outlines based on the content provided.
        #     You will receive a markdown string that contains the summary of papers and
        #     you will generate a slide outlines for each paper.
        #     Requirements:
        #     - One slide page per paper summary
        #     - Use the paper title as the slide title
        #     - Use the summary in the markdown file as the slide content,
        #       keep the key headers as bullet points but you can
        #       rephrase the content to make it more concise, and straight to the point
        #     - Each bullet point should be less than 15 words
        #     - Paragraphs of text in the slide should be less than 25 words.
        #
        #     Here is the markdown content: {summary}
        # """

        llm = new_gpt4o(0.1)
        program = FunctionCallingProgram.from_defaults(
            llm=llm,
            output_cls=SlideOutline,
            prompt_template_str=SUMMARY2OUTLINE_PMT,
            verbose=True,
        )
        response = await program.acall(
            summary=ev.summary,
            description="Data model for the slide page outline",
        )
        # response = await llm.acomplete(prompt.format(summary=ev.summary))
        return OutlineEvent(outline=response)

    @step(pass_context=True)
    async def outlines_with_layout(self, ctx: Context, ev: OutlineEvent) -> OutlinesWithLayoutEvent:
        ready = ctx.collect_events(ev, [OutlineEvent] * ctx.data["n_summaries"])
        if ready is None:
            return None
        # get slide layouts
        all_layouts = get_all_layouts_info(ctx.data["ppt_template_path"])
        all_layout_names = [layout["layout_name"] for layout in all_layouts]
        ctx.data["available_slide_layouts"] = all_layouts

        # layout_tool = FunctionTool.from_defaults(fn=get_all_layouts_info)
        # agent_worker = FunctionCallingAgentWorker.from_tools(
        #     tools=[layout_tool],
        #     llm=llm_gpt4o,
        #     allow_parallel_tool_calls=False,
        #     system_prompt=AUGMENT_LAYOUT_PMT,
        #     verbose=True
        # )
        # agent = agent_worker.as_agent()

        # add layout to outline
        llm = new_gpt4o(0.1)
        program = FunctionCallingProgram.from_defaults(
            llm=llm,
            output_cls=SlideOutlineWithLayout,
            prompt_template_str=AUGMENT_LAYOUT_PMT,
            verbose=True,
        )
        slides_w_layout = []
        for n, ev in enumerate(ready):
            response = await program.acall(
                slide_content=ev.outline.json(),
                available_layout_names=all_layout_names,
                available_layouts=all_layouts,
                description="Data model for the slide page outline with layout",
            )
            slides_w_layout.append(response)
        # response = await program.acall(
        #     summary=ev.summary,
        #     description="Data model for the slide page outline",
        # )

        return OutlinesWithLayoutEvent(outline_w_layout=slides_w_layout)

    @step(pass_context=True)
    async def outlines2code(self, ctx: Context, ev: OutlinesWithLayoutEvent) -> PythonCodeEvent:
        """Given a list of slide page outlines, generate python-pptx code to create the slides."""
        # available_layouts = ctx.data["available_slide_layouts"]
        llm = new_gpt4o(0.0)
        response = await llm.acomplete(LAYOUT2CODE_PMT.format(layout_info=ctx.data["available_slide_layouts"],
                                                              slide_content=[o.json() for o in ev.outline_w_layout]))
        return PythonCodeEvent(code=response.text)

    # @step(pass_context=True)
    # async def consolidate_outlines(self, ctx: Context, ev: OutlineEvent) -> ConsolidatedOutlineEvent:
    #     ready = ctx.collect_events(ev, [OutlineEvent] * ctx.data["n_summaries"])
    #     if ready is None:
    #         return None
    #     prompt = """"
    #     You are an AI that consolidate pieces of slide content to a full slide outline.
    #     You will receive a list of content items. Merge them to one outline with following requirements:
    #     - One slide page per item you received
    #     - Keep the individual content as is
    #
    #     The slide content is:
    #     {slide_content}
    #     """
    #     llm = new_gpt4o(0.1)
    #
    #     contents = ""
    #     for n, ev in enumerate(ready):
    #         contents += f"Slide Page {n}:\n{ev.outline}\n-----------------------------------"
    #     response = await llm.acomplete(prompt.format(slide_content=contents))
    #     return ConsolidatedOutlineEvent(outline=response.text)

    # @step(pass_context=True)
    # def outline2code(self, ctx: Context, ev: ConsolidatedOutlineEvent) -> PythonCodeEvent:
    #     ppt_template_path = ctx.data["ppt_template_path"]
    #     prompt = """"
    #     You are an AI that generate python-pptx code from a given slides outline and uses the
    #     template file provided. Return ONLY executable code as string,
    #     NEVER wrap it in a markdown code block. Do not provide explanations.
    #     The path of pptx template file is `{ppt_template_path}`.
    #     The outline of the slides is:
    #     {slide_outlines}
    #     """
    #     # user_prompt = """
    #     # Please create python-pptx code for following slide outlines with template {ppt_template_path}.
    #     # slide outlines:
    #     # {slide_outlines}
    #     # """
    #     llm = new_gpt4o(0.1)
    #     response = llm.chat(prompt.format(ppt_template_path=ppt_template_path, slide_outlines=ev.outline))
    #     return PythonCodeEvent(code=response.text)

    @step(pass_context=True)
    # def execute_code(self, ctx: Context, ev: ConsolidatedOutlineEvent) -> StopEvent:
    def execute_code(self, ctx: Context, ev: PythonCodeEvent) -> StopEvent:

        from llama_index.tools.azure_code_interpreter import (
            AzureCodeInterpreterToolSpec,
        )
        user_prompt = """
        Please create slide deck for following slide outlines with template {ppt_template_path}.
        Save the final slide deck to /mnt/data/{final_pptx_fname}.
        slide outlines:
        {slide_outlines}
        """

        from llama_index.core.agent import FunctionCallingAgentWorker
        ppt_template_path = ctx.data["ppt_template_path"]
        final_pptx_fname = ctx.data["final_pptx_fname"]

        # Create the ReActAgent and inject the tools defined in the AzureDynamicSessionsToolSpec
        layout_tool = FunctionTool.from_defaults(fn=get_all_layouts_info)

        azure_code_interpreter_spec = AzureCodeInterpreterToolSpec(
            pool_management_endpoint=settings.AZURE_DYNAMIC_SESSION_MGMT_ENDPOINT,
            local_save_path="./code_interpreter",
        )
        spec_functions = ["code_interpreter", "list_files", "upload_file"]
        azure_code_interpreter_spec.spec_functions = spec_functions

        agent = ReActAgent.from_tools(
            tools=azure_code_interpreter_spec.to_tool_list() + [layout_tool],
            llm=new_gpt4o(0.2),
            verbose=True,
            max_iterations=50
        )

        prompt = SLIDE_GEN_PMT.format(template_fpath=ppt_template_path) + REACT_PROMPT_SUFFIX
        agent.update_prompts({"agent_worker:system_prompt": PromptTemplate(prompt)})

        res = azure_code_interpreter_spec.upload_file(
            local_file_path=ppt_template_path
        )
        logging.info(f"Uploaded file to Azure: {res}")

        response = agent.chat(ev.code)

        remote_files = azure_code_interpreter_spec.list_files()
        for f in remote_files:
            logging.info(f"Downloading remote file: {f.file_full_path}")
            azure_code_interpreter_spec.download_file_to_local(
                # remote_file_path=f"/mnt/{Path(final_pptx_fname).name}",
                # local_file_path=f"code_interpreter/{Path(final_pptx_fname).name}",
                remote_file_path=f.file_full_path,
                local_file_path=f"code_interpreter/{f.filename}",
            )
        return StopEvent()


async def run_workflow(file_dir: str):
    wf = SlideGenWorkflow(
        timeout=1200, verbose=True)
    result = await wf.run(
        file_dir=file_dir,
        ppt_template_path="./data/Inmeta Brand guidelines 2023.pptx",
        final_pptx_fname="paper_summaries.pptx"
    )
    print(result)


@click.command()
@click.option("--file_dir", "-d", required=False,
              help="Path to the directory that contains paper summaries for generating slide outlines",
              default="./data/summaries_test")
def main(file_dir: str):
    asyncio.run(run_workflow(file_dir))


if __name__ == "__main__":
    draw_all_possible_flows(SlideGenWorkflow, filename="slide_gen_flows.html")
    main()
