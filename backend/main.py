from fastapi import FastAPI, HTTPException, Body
import asyncio
from fastapi.responses import StreamingResponse
import uuid
import json
import logging
from models import SlideGenFileDirectory, ResearchTopic
from workflows.slide_gen import SlideGenerationWorkflow
from workflows.summarize_and_generate_slides import SummaryAndSlideGenerationWorkflow
from workflows.summary_gen import SummaryGenerationWorkflow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI()
workflows = {}  # Store the workflow instances in a dictionary


@app.post("/run-slide-gen")
async def run_workflow_endpoint(topic: ResearchTopic):
    workflow_id = str(uuid.uuid4())

    wf = SummaryAndSlideGenerationWorkflow(timeout=2000, verbose=True)
    wf.add_workflows(
        summary_gen_wf=SummaryGenerationWorkflow(timeout=800, verbose=True)
    )
    wf.add_workflows(slide_gen_wf=SlideGenerationWorkflow(timeout=1200, verbose=True))

    # wf = SlideGenerationWorkflow(timeout=1200, verbose=True)

    workflows[workflow_id] = wf  # Store the workflow instance

    async def event_generator():
        loop = asyncio.get_running_loop()
        logging.info(f"event_generator: loop id {id(loop)}")
        yield f"{json.dumps({'workflow_id': workflow_id})}\n\n"
        task = asyncio.create_task(wf.run(user_query=topic.query))
        logging.info(f"event_generator: Created task {task}")
        try:
            async for ev in wf.stream_events():
                logging.info(f"Sending message to frontend: {ev.msg}")
                yield f"{ev.msg}\n\n"
                await asyncio.sleep(0.1)  # Small sleep to ensure proper chunking
            final_result = await task
            # yield f"[Final result]: {final_result}\n\n"
            yield f"{json.dumps({'final_result': final_result})}\n\n"
        except Exception as e:
            error_message = f"Error in workflow: {str(e)}"
            logging.error(error_message)
            yield f"{json.dumps({'event': 'error', 'message': error_message})}\n\n"
        finally:
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
        logging.info(f"submit_user_input: wf.user_input_future loop id {id(loop)}")
        if not wf.user_input_future.done():
            loop.call_soon_threadsafe(wf.user_input_future.set_result, user_input)
            logging.info("submit_user_input: set_result called")
        else:
            logging.info("submit_user_input: future already done")
        return {"status": "input received"}
    else:
        raise HTTPException(status_code=404, detail="Workflow not found or future not initialized")


@app.get("/")
async def read_root():
    return {"Hello": "World"}
