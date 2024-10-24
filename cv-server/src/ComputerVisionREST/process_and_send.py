import uvicorn
import json
from fastapi import FastAPI
from pydantic import BaseModel
from ..Base64.Base64Conversions import *
from ..PoseEstimation.PoseEstimation import PoseLandmarkExtractor

pose_landmark_extractor = PoseLandmarkExtractor()
class Frame(BaseModel):
    image_base64: str

class Preset(BaseModel):
    detection_confidence: float
    tracking_confidence: float

app = FastAPI()

@app.post("/process")
async def process_frame(frame: Frame):
    image = base64_to_image(frame.image_base64)
    landmarks_array = pose_landmark_extractor.extract_landmarks(image)
    landmarks_json = json.dumps(landmarks_array.tolist())
    return {
        "positions": json.dumps(landmarks_json)
    }

@app.post("/settings")
async def settings(preset: Preset):
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