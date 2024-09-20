import sys

from fastapi import FastAPI, HTTPException, Body
import asyncio
from fastapi.responses import StreamingResponse
import uuid
import json
import logging
from models import SlideGenFileDirectory, ResearchTopic
from workflows.slide_gen import SlideGenerationWorkflow
from workflows.summarize_and_generate_slides import SummaryAndSlideGenerationWorkflow
from workflows.summary_gen import (
    SummaryGenerationWorkflow,
    SummaryGenerationDummyWorkflow,
)
from fastapi.middleware.cors import CORSMiddleware

import mlflow
from config import settings
import os
from fastapi.responses import FileResponse
from pathlib import Path

os.environ["MLFLOW_DEFAULT_ARTIFACT_ROOT"] = "/mlruns"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Add handler and formatter if not already configured
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

app = FastAPI()
workflows = {}  # Store the workflow instances in a dictionary


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://frontend:8501"],  # Replace with your Streamlit frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/run-slide-gen")
async def run_workflow_endpoint(topic: ResearchTopic):
    workflow_id = str(uuid.uuid4())

    wf = SummaryAndSlideGenerationWorkflow(wid=workflow_id, timeout=2000, verbose=True)
    wf.add_workflows(
        summary_gen_wf=SummaryGenerationWorkflow(
            wid=workflow_id, timeout=800, verbose=True
        )
    )
    # wf.add_workflows(
    #     summary_gen_wf=SummaryGenerationDummyWorkflow(wid=workflow_id,timeout=800, verbose=True)
    # )
    wf.add_workflows(
        slide_gen_wf=SlideGenerationWorkflow(
            wid=workflow_id, timeout=1200, verbose=True
        )
    )

    # wf = SlideGenerationWorkflow(timeout=1200, verbose=True)

    workflows[workflow_id] = wf  # Store the workflow instance

    async def event_generator():
        loop = asyncio.get_running_loop()
        logger.debug(f"event_generator: loop id {id(loop)}")
        yield f"{json.dumps({'workflow_id': workflow_id})}\n\n"

        # mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        # mlflow.set_experiment("SummaryAndSlideGenerationWorkflow")
        # mlflow.llama_index.autolog()
        # mlflow.config.enable_async_logging()
        # mlflow.tracing.enable()
        # mlflow.start_run()
        # logger.debug("run_workflow_endpoint: mlflow.start_run()")

        task = asyncio.create_task(wf.run(user_query=topic.query))
        logger.debug(f"event_generator: Created task {task}")
        try:
            async for ev in wf.stream_events():
                logger.info(f"Sending message to frontend: {ev.msg}")
                yield f"{ev.msg}\n\n"
                await asyncio.sleep(0.1)  # Small sleep to ensure proper chunking
            final_result = await task

            # Construct the download URL
            download_pptx_url = f"http://backend:80/download_pptx/{workflow_id}"
            download_pdf_url = f"http://backend:80/download_pdf/{workflow_id}"

            final_result_with_url = {
                "result": final_result,
                "download_pptx_url": download_pptx_url,
                "download_pdf_url": download_pdf_url,
            }

            yield f"{json.dumps({'final_result': final_result_with_url})}\n\n"
        except Exception as e:
            error_message = f"Error in workflow: {str(e)}"
            logger.error(error_message)
            yield f"{json.dumps({'event': 'error', 'message': error_message})}\n\n"
        finally:
            # mlflow.end_run()
            # logger.debug("run_workflow_endpoint: mlflow.end_run()")
            # Clean up
            workflows.pop(workflow_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/submit_user_input")
async def submit_user_input(data: dict = Body(...)):
    workflow_id = data.get("workflow_id")
    user_input = data.get("user_input")
    wf = workflows.get(workflow_id)
    if wf and wf.user_input_future:
        loop = wf.user_input_future.get_loop()  # Get the loop from the future
        logger.info(f"submit_user_input: wf.user_input_future loop id {id(loop)}")
        if not wf.user_input_future.done():
            loop.call_soon_threadsafe(wf.user_input_future.set_result, user_input)
            logger.info("submit_user_input: set_result called")
        else:
            logger.info("submit_user_input: future already done")
        return {"status": "input received"}
    else:
        raise HTTPException(
            status_code=404, detail="Workflow not found or future not initialized"
        )


@app.get("/download_pptx/{workflow_id}")
async def download_pptx(workflow_id: str):
    # Adjust the path according to your directory structure
    file_path = (
        Path(settings.WORKFLOW_ARTIFACTS_PATH)
        / "SlideGenerationWorkflow"
        / workflow_id
        / "final.pptx"
    )
    if file_path.exists():
        return FileResponse(
            path=file_path,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename=f"final.pptx",
        )
    else:
        raise HTTPException(status_code=404, detail="File not found")


@app.get("/download_pdf/{workflow_id}")
async def download_pdf(workflow_id: str):
    file_path = (
        Path(settings.WORKFLOW_ARTIFACTS_PATH)
        / "SlideGenerationWorkflow"
        / workflow_id
        / "final.pdf"
    )
    if file_path.exists():
        return FileResponse(
            path=file_path, media_type="application/pdf", filename=f"{workflow_id}.pdf"
        )
    else:
        raise HTTPException(status_code=404, detail="PDF file not found")


@app.get("/")
async def read_root():
    return {"Hello": "World"}
