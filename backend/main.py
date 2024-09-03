from fastapi import FastAPI, HTTPException
import asyncio
from fastapi.responses import StreamingResponse

from models import SlideGenFileDirectory
from workflows.slide_gen_agent import run_workflow, SlideGenWorkflow

app = FastAPI()


@app.post("/run-slide-gen")
async def run_workflow_endpoint(file_dir: SlideGenFileDirectory):
    wf = SlideGenWorkflow(timeout=1200, verbose=True)

    async def generate():
        task = asyncio.create_task(wf.run(file_dir=file_dir.path))
        async for ev in wf.stream_events():
            yield f"data: {ev.msg}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

    # def generate():
    #     streaming_response = wf.run(
    #         file_dir=file_dir.path,
    #     )
    #     for text in streaming_response.:
    #         yield f"data: {text}\n\n"
    #
    # return StreamingResponse(generate(), media_type="text/event-stream")
    # def generate():
    #     response = await run_workflow(file_dir.path)
    #     for text in response.response_gen:
    #         yield f"data: {text}\n\n"
    #
    #
    # try:
    #     response = await run_workflow(file_dir.path)
    #     async for token in response:
    #         print(token.delta, end="", flush=True)
    #     # return {"status": "Workflow completed successfully"}
    #     return StreamingResponse(generate(), media_type="text/event-stream")
    #
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str = None):
    if item_id < 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"item_id": item_id, "q": q}


@app.get("/users/{user_id}")
async def read_user(user_id: int):
    if user_id < 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user_id}


@app.get("/")
async def read_root():
    return {"Hello": "World"}
