import uvicorn
import json
import httpx
import os
import signal
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from ..Base64.Base64Conversions import *
from ..PoseEstimation.PoseEstimation import PoseLandmarkExtractor


class Frame(BaseModel):
    image_base64: str

class Preset(BaseModel):
    detection_confidence: float
    tracking_confidence: float

class PoseProcessorServer:
    def __init__(self) -> None:
        self.app = FastAPI(lifespan=self.lifespan)
        self.UNITY_SERVER_URL = "http://localhost:7000"
        self.UNITY_SHUTDOWN_ENDPOINT = "/shutdown/"
        self.IS_UNITY_RUNNING = False
        self.pose_landmark_extractor = None
        self.define_routes()

    def start_server(self) -> None:
        uvicorn.run(self.app, host="127.0.0.1", port=8000)

    def define_routes(self) -> None:
        @self.app.get("/health")
        async def health_check():
            """
            Function for unity server to check if computer vision server is running
            """
            self.IS_UNITY_RUNNING = True
            return {
                "status" : "OK"
            }

        @self.app.post("/process")
        async def process_frame(frame: Frame):
            """
            Function to recieve images in base64 format and then process them through mediapipe model
            and send joint positions in return
            """
            image = base64_to_image(frame.image_base64)
            landmarks_array = self.pose_landmark_extractor.extract_landmarks(image)
            landmarks_json = json.dumps(landmarks_array.tolist())
            return {
                "positions": json.dumps(landmarks_json)
            }

        @self.app.post("/settings")
        async def settings(preset: Preset):
            """
            Applying settings given by unity server to the mediapipe model 
            """
            self.pose_landmark_extractor = PoseLandmarkExtractor(preset.detection_confidence, preset.tracking_confidence)
            return {
                "message": "Settings applied successfully",
            }

        @self.app.post("/shutdown")
        async def shutdown():
            """
            Signal from unity server to shutdown computer vision server
            """
            self.IS_UNITY_RUNNING = False
            os.kill(os.getpid(), signal.SIGTERM)
            return {
                "message": "CVServer shutting down"
            }
    @asynccontextmanager
    async def lifespan(self, app:FastAPI):
        yield
        if self.IS_UNITY_RUNNING:
            async with httpx.AsyncClient() as client:
                try:
                    unity_response = await client.post(
                        self.UNITY_SERVER_URL + self.UNITY_SHUTDOWN_ENDPOINT,
                        json = {"message": "Computer vision server is shutting down"}
                    )
                except Exception as error:
                    print(f"Failed to send shutdown signal: {error}")
        

if __name__ == "__main__":
    pose_processor_server = PoseProcessorServer()
    pose_processor_server.start_server()