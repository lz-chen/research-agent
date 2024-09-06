from fastapi import FastAPI, HTTPException
import asyncio
from fastapi.responses import StreamingResponse

from models import SlideGenFileDirectory
from workflows.slide_gen_agent import SlideGenWorkflow
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI()


@app.post("/run-slide-gen")
async def run_workflow_endpoint(file_dir: SlideGenFileDirectory):
    wf = SlideGenWorkflow(timeout=1200, verbose=True)

    async def event_generator():
        task = asyncio.create_task(wf.run(file_dir=file_dir.path))
        async for ev in wf.stream_events():
            # print(f"backend streaming event: {ev.msg}")
            yield f"{ev.msg}\n\n"
            await asyncio.sleep(0.1)  # Small sleep to ensure proper chunking
        final_result = await task
        yield f"Final result: {final_result}\n\n"
        await asyncio.sleep(0.1)  # Ensure the final message is sent
        yield "END_OF_STREAM"  # Signal the end of the stream

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/")
async def read_root():
    return {"Hello": "World"}
