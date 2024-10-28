import uvicorn
import json
import httpx
import os
import signal
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
from ..Base64.Base64Conversions import *

UNITY_SERVER_URL = "http://localhost:7000"
UNITY_SHUTDOWN_ENDPOINT = "/shutdown/"
IS_UNITY_RUNNING = False

app = FastAPI()

class Frame(BaseModel):
    image_base64: str

@app.get("/health")
async def health_check():
    global IS_UNITY_RUNNING
    IS_UNITY_RUNNING = True
    return {
        "status": "OK"
    }

@app.post("/capture_and_send")
async def capture_and_send():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise Exception("Failed to capture frame")
    
    image_base64 = image_to_base64(frame).decode('utf-8')
    cap.release()
    
    return image_base64


@app.post("/shutdown")
async def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)
    return {"message": "Server shutting down"}

@app.on_event("shutdown")
async def shutdown_event():
    if IS_UNITY_RUNNING:
        async with httpx.AsyncClient() as client:
            try:
                await client.post(
                    UNITY_SERVER_URL + UNITY_SHUTDOWN_ENDPOINT,
                    json={"message": "Camera capture server is shutting down"}
                )
            except Exception as error:
                print(f"Failed to send shutdown signal: {error}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)