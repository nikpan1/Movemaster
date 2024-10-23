from fastapi import FastAPI
from pydantic import BaseModel

class Frame(BaseModel):
    image_base64: str

class Preset(BaseModel):
    threshold: float
    model: str

app = FastAPI()

@app.post("/process")
async def process_frame(frame: Frame):
    image = frame.image_base64
    return {"message": "Image received"}

@app.post("/settings")
async def settings(preset: Preset):
    return {"x": "y"}

@app.post("/shutdown")
async def shutdown():
    return {"xyz"}