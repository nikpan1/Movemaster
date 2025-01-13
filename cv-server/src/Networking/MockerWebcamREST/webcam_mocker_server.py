import logging
import os
import signal

import httpx
import uvicorn
from fastapi import FastAPI, Request

from Networking.Base64.base64_conversions import *
from Networking.MockerWebcamREST.mocker_video_capture import MockerVideoCapture


class MockerCaptureCameraServer:
    def __init__(self, mocker, unity_server_address="localhost", unity_port_number=7000, max_threads=10):
        self.UNITY_SERVER_URL = f"http://{unity_server_address}:{unity_port_number}"
        self.IS_UNITY_RUNNING = False

        self.current_frame = self.get_blank_image()
        self.mocker = mocker

        self.app = FastAPI()
        self.setup_calls()

    def start_server(self) -> None:
        # -------------------------------------------
        # @TODO should be parametrized in the future -- TASK-77
        cv_server_address = "127.0.0.1"
        cv_server_port = 8001
        # -------------------------------------------

        uvicorn.run(self.app, host=cv_server_address, port=cv_server_port)

    @staticmethod
    def shutdown_server() -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    @staticmethod
    def get_blank_image() -> np.ndarray:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def setup_calls(self) -> None:
        @self.app.get("/healthcheck")
        async def health_check():
            self.IS_UNITY_RUNNING = True
            return {"status": "OK"}

        @self.app.delete("/shutdown")
        async def shutdown():
            logging.info("Shutdown signal.")
            self.IS_UNITY_RUNNING = False
            self.shutdown_server()
            return {"message": "Mocker Capture Camera Server shutting down"}

        @self.app.get("/new_frame")
        async def capture_and_send():
            return {"base64_image": image_to_base64(self.mocker.current_frame),
                    "latest_predicted_class": self.mocker.latest_predicted_class,
                    "latest_predicted_confidence": self.mocker.latest_predicted_confidence}

        @self.app.get("/list_cameras")
        async def list_cameras():
            """
            Returns a list of available cameras.
            """
            cams = MockerVideoCapture.list_available_cameras(max_tested=5)
            return {
                "cameras": [
                    {"id": idx, "name": name}
                    for idx, name in cams
                ]
            }

        """@self.app.post("/set_camera")
        async def set_camera(request: Request):
            try:
                logging.info("Endpoint /set_camera called")
                body = await request.json()
                device_index = body.get("device_index", 0)
                logging.info(f"Attempting to set camera to device_index: {device_index}")
                self.cap.set(device_index)
                logging.info(f"Successfully set camera to device_index: {device_index}")
                return {"message": f"Camera set to {device_index}"}
            except Exception as e:
                logging.error(f"Error in /set_camera: {e}")
                return {"error": str(e)}, 500"""

        @self.app.on_event("shutdown")
        async def shutdown_event():
            if not self.IS_UNITY_RUNNING:
                async with httpx.AsyncClient() as client:
                    try:
                        await client.post(self.UNITY_SERVER_URL + "/shutdown/",
                                          json={"message": "Mocker Camera capture server is shutting down"})
                    except Exception as error:
                        logging.warning(f"Failed to send shutdown signal: {error}")
