import uvicorn
import httpx
import os
import signal
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from src.Base64.Base64Conversions import *


class Frame(BaseModel):
    image_base64: str


class CaptureCameraServer:
    def __init__(self) -> None:
        self.UNITY_SERVER_URL = "http://localhost:7000"
        self.UNITY_SHUTDOWN_ENDPOINT = "/shutdown/"
        self.IS_UNITY_RUNNING = False
        self.cap = cv2.VideoCapture(0)
        self.app = FastAPI()

    def start_server(self) -> None:
        self.setup_calls()
        uvicorn.run(self.app, host="127.0.0.1", port=8001)

    def shutdown_server(self) -> None:
        if self.cap.isOpened():
            self.cap.release()
        os.kill(os.getpid(), signal.SIGTERM)

    def setup_calls(self) -> None:
        @self.app.get("/health")
        async def health_check():
            self.IS_UNITY_RUNNING = True
            return {
                "status": "OK"
            }

        @self.app.post("/capture_and_send")
        async def capture_and_send():
            if not self.cap.isOpened():
                self.cap.open(0)
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Failed to capture frame")

            image_base64 = image_to_base64(frame).decode('utf-8')

            return image_base64

        @self.app.post("/shutdown")
        async def shutdown(background_tasks: BackgroundTasks):
            self.IS_UNITY_RUNNING = False
            background_tasks.add_task(self.shutdown_server)
            return {"message": "Capture Camera Server shutting down"}

        @self.app.on_event("shutdown")
        async def shutdown_event():
            if not self.IS_UNITY_RUNNING:
                async with httpx.AsyncClient() as client:
                    try:
                        await client.post(
                            self.UNITY_SERVER_URL + self.UNITY_SHUTDOWN_ENDPOINT,
                            json={"message": "Camera capture server is shutting down"}
                        )
                    except Exception as error:
                        print(f"Failed to send shutdown signal: {error}")


if __name__ == "__main__":
    webcamImageServer = CaptureCameraServer()
    webcamImageServer.start_server()

