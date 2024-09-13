from fastapi import FastAPI, HTTPException, Body
import asyncio
from fastapi.responses import StreamingResponse
import uuid
from models import SlideGenFileDirectory
from workflows.slide_gen import SlideGenerationWorkflow
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()
workflows = {}  # Store the workflow instances in a dictionary


@app.post("/run-slide-gen")
async def run_workflow_endpoint(file_dir: SlideGenFileDirectory):
    workflow_id = str(uuid.uuid4())
    wf = SlideGenerationWorkflow(timeout=1200, verbose=True)
    workflows[workflow_id] = wf  # Store the workflow instance

    async def event_generator():
        yield f'{{"workflow_id": "{workflow_id}"}}\n\n'
        task = asyncio.create_task(wf.run(file_dir=file_dir.path))
        async for ev in wf.stream_events():
            print(f"Sending message to frontend: {ev.msg}")
            yield f"{ev.msg}\n\n"
            await asyncio.sleep(0.1)  # Small sleep to ensure proper chunking
        final_result = await task
        yield f"[Final result]: {final_result}\n\n"
        # Clean up
        workflows.pop(workflow_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/submit_user_input")
async def submit_user_input(data: dict = Body(...)):
    workflow_id = data.get("workflow_id")
    user_input = data.get("user_input")
    wf = workflows.get(workflow_id)
    if wf:
        wf.user_input = user_input
        wf.user_input_event.set()
        return {"status": "input received"}
    else:
        raise HTTPException(status_code=404, detail="Workflow not found")


@app.get("/")
async def read_root():
    return {"Hello": "World"}
