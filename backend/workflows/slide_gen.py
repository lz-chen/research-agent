import asyncio
import json
import random
import string

import click

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.program import (
    FunctionCallingProgram,
    MultiModalLLMCompletionProgram,
)
from llama_index.core.tools import FunctionTool

from config import settings
from prompts.prompts import (
    SLIDE_GEN_PMT,
    REACT_PROMPT_SUFFIX,
    SUMMARY2OUTLINE_PMT,
    AUGMENT_LAYOUT_PMT,
    SLIDE_VALIDATION_PMT,
    SLIDE_MODIFICATION_PMT,
    MODIFY_SUMMARY2OUTLINE_PMT,
)
from services.llms import llm_gpt4o, new_gpt4o, new_gpt4o_mini, mm_gpt4o
from services.embeddings import aoai_embedder
import logging
import sys
from llama_index.core import PromptTemplate
from llama_index.core.workflow import (
    Context,
    StartEvent,
    StopEvent,
    Workflow,
    step,
    draw_all_possible_flows,
)

from utils.tools import get_all_layouts_info
import inspect
from llama_index.tools.azure_code_interpreter import (
    AzureCodeInterpreterToolSpec,
)

from utils.file_processing import pptx2images
from workflows.events import *
import mlflow

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


class SlideGenerationWorkflow(Workflow):
    max_validation_retries: int = 5
    slide_template_path: str = "./data/Inmeta 2023 template.pptx"
    final_slide_fname: str = "paper_summaries.pptx"
    slide_outlines_fname: str = "slide_outlines.json"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # make random string of length 10 and make it a suffix for WORKFLOW_ARTIFACTS_PATH
        class_name = self.__class__.__name__
        s = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        self.workflow_artifacts_path = (
            Path(settings.WORKFLOW_ARTIFACTS_PATH).joinpath(class_name).joinpath(s)
        )
        self.workflow_artifacts_path.mkdir(parents=True, exist_ok=True)

        self.azure_code_interpreter = AzureCodeInterpreterToolSpec(
            pool_management_endpoint=settings.AZURE_DYNAMIC_SESSION_MGMT_ENDPOINT,
            local_save_path=self.workflow_artifacts_path.as_posix(),
        )
        spec_functions = ["code_interpreter", "list_files", "upload_file"]
        self.azure_code_interpreter.spec_functions = spec_functions
        self.pdf2images_tool = FunctionTool.from_defaults(fn=pptx2images)
        self.save_python_code_tool = FunctionTool.from_defaults(
            fn=self.save_python_code
        )
        self.all_layout = get_all_layouts_info(self.slide_template_path)
        self.all_layout_tool = FunctionTool.from_defaults(fn=self.get_all_layout)

    def get_all_layout(self):
        """Get all layout information"""
        return self.all_layout

    def download_all_files_from_session(self):
        """Download all files from the Azure session"""
        remote_files = self.azure_code_interpreter.list_files()
        local_files = []
        for f in remote_files:
            logging.info(f"Downloading remote file: {f.file_full_path}")
            local_path = f"{self.workflow_artifacts_path.as_posix()}/{f.filename}"
            self.azure_code_interpreter.download_file_to_local(
                remote_file_path=f.file_full_path,
                local_file_path=local_path,
            )
            local_files.append(local_path)
        return local_files

    def save_python_code(self, code: str):
        """Save the python code to file"""
        with open(f"{self.workflow_artifacts_path}/code.py", "w") as f:
            f.write(code)

    @staticmethod
    def _agent_thought_to_stream(ctx, agent_task):
        cur_reasonings = agent_task.extra_state["current_reasoning"]
        for r in cur_reasonings:
            ctx.write_event_to_stream(
                Event(
                    msg=f"[{inspect.currentframe().f_code.co_name}] React Agent Reasoning: {r.get_content()}"
                )
            )

    def run_react_agent(self, agent, task, ctx):
        step_output = agent.run_step(task.task_id)
        self._agent_thought_to_stream(ctx, task)
        while not step_output.is_last:
            step_output = agent.run_step(task.task_id)
            self._agent_thought_to_stream(ctx, task)
        response = agent.finalize_response(task.task_id)
        ctx.write_event_to_stream(
            Event(
                msg=f"[{inspect.currentframe().f_code.co_name}] React Agent Final Response: {response}"
            )
        )

    @step(pass_context=True, num_workers=4)
    def get_summaries(self, ctx: Context, ev: StartEvent) -> SummaryEvent:
        """Entry point of the workflow. Read the content of the summary files from provided
        directory. For each summary file, send a SummaryEvent to the next step."""
        ctx.write_event_to_stream(
            Event(
                msg=f"[{inspect.currentframe().f_code.co_name}] Reading summaries from markdown files..."
            )
        )
        ctx.data["n_retry"] = 0  # keep count of slide validation retries
        markdown_files = list(Path(ev.get("file_dir")).glob("*.md"))
        ctx.data["n_summaries"] = len(
            markdown_files
        )  # make sure later step collect all the summaries
        for i, f in enumerate(markdown_files):
            s = read_summary_content(f)
            ctx.write_event_to_stream(
                Event(
                    msg=f"[{inspect.currentframe().f_code.co_name}] Sending {i}th summaries..."
                )
            )
            self.send_event(SummaryEvent(summary=s))

    @step(pass_context=True, num_workers=4)
    async def summary2outline(
            self, ctx: Context, ev: SummaryEvent | OutlineFeedbackEvent
    ) -> OutlineEvent:
        """Convert the summary content of one paper to slide outline of one page, mainly
        condense and shorten the elaborated summary content to short sentences or bullet points.
        """
        ctx.write_event_to_stream(
            Event(
                msg=f"[{inspect.currentframe().f_code.co_name}] Making summary to slide outline..."
            )
        )
        llm = new_gpt4o_mini(0.1)
        if isinstance(ev, OutlineFeedbackEvent):
            program = FunctionCallingProgram.from_defaults(
                llm=llm,
                output_cls=SlideOutline,
                prompt_template_str=MODIFY_SUMMARY2OUTLINE_PMT,
                verbose=True,
            )
            response = await program.acall(
                summary_txt=ev.summary,
                outline_txt=ev.outline.json(),
                feedback=ev.feedback,
                description="Data model for the slide page outline",
            )

        else:
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
        # async for response in generator:
        #     # Allow the workflow to stream this piece of response
        json_resp = {"original_summary": ev.summary}
        json_resp.update(json.loads(response.json()))

        ctx.write_event_to_stream(
            Event(
                msg=f"[{inspect.currentframe().f_code.co_name}]/json: {json.dumps(json_resp)}"
            )
        )
        return OutlineEvent(summary=ev.summary, outline=response)

    @step(pass_context=True)
    async def gather_feedback_outline(
            self, ctx: Context, ev: OutlineEvent
    ) -> OutlineFeedbackEvent | OutlineOkEvent:
        """Present user the original paper summary and the outlines generated, gather feedback from user"""
        # ready = ctx.collect_events(ev, [OutlineEvent] * ctx.data["n_summaries"])
        ctx.write_event_to_stream(
            Event(
                msg=f"[{inspect.currentframe().f_code.co_name}] Gathering feedback on the outline..."
            )
        )

        print(f"the original summary is: {ev.summary}")
        print(f"the outline is: {ev.outline}")
        print("Do you want to proceed with this outline? (yes/no):")
        # feedback = input()
        feedback = "yes"
        if feedback.lower().strip() in ["yes", "y"]:
            return OutlineOkEvent(summary=ev.summary, outline=ev.outline)
        else:
            print("Please provide feedback on the outline:")
            # feedback = input()
            feedback = "The outline is too verbose, please make it more concise."
            return OutlineFeedbackEvent(
                summary=ev.summary, outline=ev.outline, feedback=feedback
            )

    @step(pass_context=True)
    async def outlines_with_layout(
            self, ctx: Context, ev: OutlineOkEvent
    ) -> OutlinesWithLayoutEvent:
        """Given a list of slide page outlines, augment each outline with layout information.
        The layout information includes the layout name, the index of the title placeholder,
        and the index of the content placeholder. Return an event with the augmented outlines.
        """
        ready = ctx.collect_events(ev, [OutlineOkEvent] * ctx.data["n_summaries"])
        if ready is None:
            return None
        ctx.write_event_to_stream(
            Event(
                msg=f"[{inspect.currentframe().f_code.co_name}] Outlines for all paper is ready! Adding layout info..."
            )
        )
        all_layout_names = [layout["layout_name"] for layout in self.all_layout]

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
                available_layouts=self.all_layout,
                description="Data model for the slide page outline with layout",
            )
            slides_w_layout.append(response)

        # store the slide outlines as json file
        slide_outlines_json = self.workflow_artifacts_path.joinpath(
            self.slide_outlines_fname
        )
        with slide_outlines_json.open("w") as f:
            json.dump([o.json() for o in slides_w_layout], f, indent=4)
        # ctx.data["slide_outlines_json"] = slide_outlines_json
        ctx.write_event_to_stream(
            Event(
                msg=f"[{inspect.currentframe().f_code.co_name}] {len(slides_w_layout)} outlines with layout are ready!"
                    f" Stored in {slide_outlines_json}"
            )
        )

        return OutlinesWithLayoutEvent(
            outlines_fpath=slide_outlines_json, outline_example=slides_w_layout[0]
        )

    @step(pass_context=True)
    async def slide_gen(
            self, ctx: Context, ev: OutlinesWithLayoutEvent
    ) -> SlideGeneratedEvent:
        agent = ReActAgent.from_tools(
            tools=self.azure_code_interpreter.to_tool_list() + [self.all_layout_tool],
            llm=new_gpt4o(0.1),
            verbose=True,
            max_iterations=50,
        )

        prompt = (
                SLIDE_GEN_PMT.format(
                    json_file_path=ev.outlines_fpath.as_posix(),
                    template_fpath=self.slide_template_path,
                    final_slide_fname=self.final_slide_fname,
                )
                + REACT_PROMPT_SUFFIX
        )
        agent.update_prompts({"agent_worker:system_prompt": PromptTemplate(prompt)})

        res = self.azure_code_interpreter.upload_file(
            local_file_path=self.slide_template_path
        )
        logging.info(f"Uploaded file to Azure: {res}")

        task = agent.create_task(
            f"An example of outline item in json is {ev.outline_example.json()},"
            f" generate a slide deck"
        )
        self.run_react_agent(agent, task, ctx)
        local_files = self.download_all_files_from_session()
        ctx.write_event_to_stream(
            Event(
                msg=f"[{inspect.currentframe().f_code.co_name}] Downloaded files to local path: {local_files}"
            )
        )
        return SlideGeneratedEvent(
            pptx_fpath=f"{self.workflow_artifacts_path}/{self.final_slide_fname}"
        )

    @step(pass_context=True)
    async def validate_slides(
            self, ctx: Context, ev: SlideGeneratedEvent
    ) -> StopEvent | SlideValidationEvent:
        """Validate the generated slide deck"""
        ctx.data["n_retry"] += 1
        ctx.data["latest_pptx_file"] = Path(ev.pptx_fpath).name
        logging.info(f"[validate_slides] the retry count is: {ctx.data['n_retry']}")
        logging.info(
            f"[validate_slides] Validating the generated slide deck"
            f" {Path(ev.pptx_fpath).name} | {ctx.data['latest_pptx_file']}...")
        # slide to images
        img_dir = pptx2images(Path(ev.pptx_fpath))

        # upload image w. prompt for validation to llm and get structured response
        image_documents = SimpleDirectoryReader(img_dir).load_data()

        ctx.write_event_to_stream(
            Event(
                msg=f"[{inspect.currentframe().f_code.co_name}] "
                    f"{ctx.data['n_retry']}th try for validating the generated slide deck..."
            )
        )

        llm = mm_gpt4o
        program = MultiModalLLMCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(SlideValidationResult),
            image_documents=image_documents,
            prompt_template_str=SLIDE_VALIDATION_PMT,
            multi_modal_llm=llm,
            verbose=True,
        )
        response = program()

        if response.is_valid:
            ctx.write_event_to_stream(
                Event(
                    msg=f"[{inspect.currentframe().f_code.co_name}] The slides are fixed!"
                )
            )
            return StopEvent(
                self.workflow_artifacts_path.joinpath(self.final_slide_fname)
            )
        else:
            if ctx.data["n_retry"] < self.max_validation_retries:
                ctx.write_event_to_stream(
                    Event(
                        msg=f"[{inspect.currentframe().f_code.co_name}] The slides are not fixed, retrying..."
                    )
                )
                return SlideValidationEvent(result=response)
            else:
                ctx.write_event_to_stream(
                    Event(
                        msg=f"[{inspect.currentframe().f_code.co_name}] "
                            f"The slides are not fixed after {self.max_validation_retries} retries!"
                    )
                )
                return StopEvent(
                    f"The slides are not fixed after {self.max_validation_retries} retries!"
                )

    @step(pass_context=True)
    async def modify_slides(
            self, ctx: Context, ev: SlideValidationEvent
    ) -> SlideGeneratedEvent:
        """Modify the slides based on the validation feedback"""

        # give agent code_interpreter and get_layout tools
        # use feedback as prompt to agent
        # agent make changes to the slides and save slide
        ctx.write_event_to_stream(
            Event(
                msg=f"[{inspect.currentframe().f_code.co_name}] Modifying the slides based on the feedback..."
            )
        )

        slide_pptx_path = f"/mnt/data/{ctx.data['latest_pptx_file']}"
        logging.info(f"[modify_slides] slide_pptx_path: {slide_pptx_path}")
        remote_files = self.azure_code_interpreter.list_files()
        for f in remote_files:
            if f.filename == self.final_slide_fname:
                slide_pptx_path = f.file_full_path
        modified_pptx_path = f"{Path(slide_pptx_path).stem}_v{ctx.data['n_retry']}.pptx"
        logging.info(f"[modify_slides] modified_pptx_path: {modified_pptx_path}")

        agent = ReActAgent.from_tools(
            tools=self.azure_code_interpreter.to_tool_list() + [self.all_layout_tool],
            llm=new_gpt4o(0.1),
            verbose=True,
            max_iterations=50,
        )
        prompt = SLIDE_MODIFICATION_PMT + REACT_PROMPT_SUFFIX

        agent.update_prompts({"agent_worker:system_prompt": PromptTemplate(prompt)})
        # # response = agent.chat(f"Modify the slides based on the feedback")
        task = agent.create_task(f"The latest version of the slide deck can be found here: "
                                 f"`/mnt/data/{ctx.data['latest_pptx_file']}`.\n"
                                 f"The feedback provided is as follows: '{ev.result.suggestion_to_fix}'\n"
                                 f"Save the modified slide deck as `{modified_pptx_path}`.")
        self.run_react_agent(agent, task, ctx)

        self.download_all_files_from_session()
        return SlideGeneratedEvent(
            pptx_fpath=f"{self.workflow_artifacts_path.as_posix()}/{Path(modified_pptx_path).name}"
        )


async def run_workflow(file_dir: str):
    wf = SlideGenerationWorkflow(timeout=1200, verbose=True)
    result = await wf.run(
        file_dir=file_dir,
    )
    print(result)


@click.command()
@click.option(
    "--file_dir",
    "-d",
    required=False,
    help="Path to the directory that contains paper summaries for generating slide outlines",
    default="./data/summaries_test",
)
def main(file_dir: str):
    draw_all_possible_flows(SlideGenerationWorkflow, filename="slide_gen_flows.html")
    mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("research-agent-slide-gen-wf")
    mlflow.llama_index.autolog()
    mlflow.start_run()

    asyncio.run(run_workflow(file_dir))

    mlflow.end_run()


if __name__ == "__main__":
    main()
