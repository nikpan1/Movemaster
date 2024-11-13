import json
import logging
import os
import signal
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI

from ComputerVision.PoseEstimation.PoseEstimation import PoseLandmarkExtractor
from Networking.Base64.base64_conversions import *
from Networking.ComputerVisionREST.models import *


class ComputerVisionServer:
    def __init__(self, unity_server_address="localhost", unity_port_number=7000):
        self.UNITY_SERVER_URL = f"http://{unity_server_address}:{unity_port_number}"

        self.IS_UNITY_RUNNING = False
        self.pose_landmark_extractor = None

        self.app = FastAPI(lifespan=self.lifespan)
        self.setup_calls()

    def start_server(self) -> None:
        # -------------------------------------------
        # @TODO should be parametrized in the future -- TASK-77
        cv_server_address = "127.0.0.1"
        cv_server_port = 8000
        # -------------------------------------------

        uvicorn.run(self.app, host=cv_server_address, port=cv_server_port)

    def setup_calls(self) -> None:
        @self.app.get("/health")
        async def health_check():
            """
            Function for unity server to check if computer vision server is running
            """
            self.IS_UNITY_RUNNING = True
            return {"status": "OK"}

        @self.app.post("/process")
        async def process_frame(frame: Frame):
            """
            Function to receive images in base64 format and then process them through estimation model
            and return its result
            """
            if self.pose_landmark_extractor is None:
                logging.warning("pose_landmark_extractor is None.")
                return {"positions": json.dumps([])}

            image = base64_to_image(frame.image_base64)
            landmarks_array = self.pose_landmark_extractor.extract_landmarks(image)
            landmarks_json = json.dumps(landmarks_array.tolist())
            return {"positions": json.dumps(landmarks_json)}

        @self.app.post("/settings")
        async def load_settings(settings: Settings):
            """
            Applying settings given by unity server to pose estimation algorithm
            """
            self.pose_landmark_extractor = PoseLandmarkExtractor(settings.detection_confidence,
                                                                 settings.tracking_confidence)
            return {"message": "Settings applied successfully"}

        @self.app.post("/shutdown")
        async def shutdown():
            """
            Signal from unity server to shut down computer vision server
            """
            self.IS_UNITY_RUNNING = False
            os.kill(os.getpid(), signal.SIGTERM)
            return {"message": "CVServer shutting down"}

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """
        Handles termination by sending a shutdown signal to Unity server if running.
        """
        yield
        if self.IS_UNITY_RUNNING:
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(f"{self.UNITY_SERVER_URL}/shutdown/",
                                                 json={"message": "CVServer is shutting down"})

                    # Raises an error for non-2xx responses
                    response.raise_for_status()

                except httpx.RequestError as req_err:
                    logging.warning(f"Network error while sending shutdown signal: {req_err}")

                except httpx.HTTPStatusError as http_err:
                    logging.warning(f"HTTP error {http_err.response.status_code} on shutdown signal: {http_err}")

                except Exception as error:
                    logging.warning(f"Unexpected error on shutdown signal: {error}")

                finally:
                    logging.info("Shutdown signal successfully sent to Unity server.")
