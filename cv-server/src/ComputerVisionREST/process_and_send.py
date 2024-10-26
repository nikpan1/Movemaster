import uvicorn
import json
import httpx
import os
import signal
from fastapi import FastAPI
from pydantic import BaseModel
from ..Base64.Base64Conversions import *
from ..PoseEstimation.PoseEstimation import PoseLandmarkExtractor

UNITY_SERVER_URL = "http://localhost:7000"
UNITY_SHUTDOWN_ENDPOINT = "/shutdown/"
IS_UNITY_RUNNING = False


app = FastAPI()

pose_landmark_extractor = PoseLandmarkExtractor()

class Frame(BaseModel):
    image_base64: str

class Preset(BaseModel):
    detection_confidence: float
    tracking_confidence: float

@app.get("/health")
async def health_check():
    IS_UNITY_RUNNING = True
    return {
        "status" : "OK"
    }

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
    os.kill(os.getpid(), signal.SIGTERM)
    return FastAPI.Response(status_code=200)

@app.on_event("shutdown")
async def shutdown_event():
    if IS_UNITY_RUNNING:
        async with httpx.AsyncClient() as client:
            try:
                unity_response = await client.post(
                    UNITY_SERVER_URL + UNITY_SHUTDOWN_ENDPOINT,
                    json = {"message": "Computer vision server is shutting down"}
                )
            except Exception as error:
                print(f"Failed to send shutdown signal: {error}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)