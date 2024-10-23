import uvicorn
from fastapi import FastAPI
from numpy.ma.core import minimum
from pydantic import BaseModel
from ..Base64.Base64Conversions import *
from ..PoseEstimation.PoseEstimation import PoseLandmarkExtractor

pose_landmark_extractor = PoseLandmarkExtractor()

class Frame(BaseModel):
    image_base64: str

class Preset(BaseModel):
    threshold: float
    model: str

app = FastAPI()

@app.post("/process")
async def process_frame(frame: Frame):
    image = base64_to_image(frame.image_base64)
    landmarks_array = pose_landmark_extractor.extract_landmarks(image)
    return {
        "positions": landmarks_array
    }

@app.post("/settings")
async def settings(preset: Preset):
    # minimum tracking confidence instead of model
    pose_landmark_extractor = PoseLandmarkExtractor(preset.threshold, preset.model)
    return {
        "message": "Settings applied successfully",
    }

@app.post("/shutdown")
async def shutdown():
    return {
        "Goodbye"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)