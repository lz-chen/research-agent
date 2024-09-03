from fastapi import FastAPI, HTTPException

app = FastAPI()
import asyncio
from workflows.slide_gen_agent import run_workflow


@app.post("/run-slide-gen/")
async def run_workflow_endpoint(file_dir: str):
    try:
        asyncio.run(run_workflow(file_dir))
        return {"status": "Workflow completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}
from fastapi import FastAPI

app = FastAPI()

@app.post("/run-slide-gen")
async def run_slide_gen(file_dir: str):
    # Placeholder logic for slide generation
    return {"message": f"Slides will be generated for the directory: {file_dir}"}
