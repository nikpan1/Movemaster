import os
import signal
import httpx
import uvicorn
import logging

from fastapi import FastAPI, BackgroundTasks

from Networking.Base64.base64_conversions import *
from Networking.WebcamREST.webcam_video_capture import WebcamVideoCapture
import asyncio

class WebcamCaptureServer:
    def __init__(self, unity_server_address="localhost", unity_port_number=7000, device_id=0):
        self.UNITY_SERVER_URL = f"http://{unity_server_address}:{unity_port_number}"

        self.IS_UNITY_RUNNING = False

        self.cap = WebcamVideoCapture()
        self.cap.set(device_id)

        self.app = FastAPI()
        self.setup_calls()

    def start_server(self) -> None:
        # -------------------------------------------
        # @TODO should be parametrized in the future -- TASK-77
        cv_server_address = "127.0.0.1"
        cv_server_port = 8001
        # -------------------------------------------

        uvicorn.run(self.app, host=cv_server_address, port=cv_server_port)

    def shutdown_server(self) -> None:
        self.cap.reset()
        os.kill(os.getpid(), signal.SIGTERM)

    def setup_calls(self) -> None:
        @self.app.get("/health")
        async def health_check():
            self.IS_UNITY_RUNNING = True
            return {"status": "OK"}

        @self.app.post("/capture_and_send")
        async def capture_and_send():
            ret, frame = self.cap.get_frame()
            if not ret:
                raise Exception("Failed to capture frame")

            return image_to_base64(frame)

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
                        await client.post(self.UNITY_SERVER_URL + "/shutdown/",
                                          json={"message": "Camera capture server is shutting down"})
                    except Exception as error:
                        logging.warning(f"Failed to send shutdown signal: {error}")

    def send_frame_to_unity(self, frame_base64: str) -> None:
        """
        Sends the provided base64-encoded frame as a plain string to the Unity server using a POST request to '/new_frame',
        ensuring proper task management to avoid warnings or errors.
        :param frame_base64: Base64-encoded string representing the frame
        """
        async def _send():
            try:
                unity_endpoint = f"{self.UNITY_SERVER_URL}/new_frame"
                async with httpx.AsyncClient() as client:
                    await client.post(unity_endpoint, data=frame_base64, headers={"Content-Type": "text/plain"})
            except Exception as e:
                logging.error(f"Error sending frame to Unity: {e}")

        try:
            # Get the current running event loop
            loop = asyncio.get_running_loop()
            loop.create_task(_send())  # Schedule the coroutine
        except RuntimeError:
            # If no loop is running, create one explicitly and run the task
            asyncio.run(_send())