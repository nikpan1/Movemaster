import os
import signal
import httpx
import uvicorn
import logging

from fastapi import FastAPI

from Networking.Base64.base64_conversions import *


class MockerCaptureCameraServer:
    def __init__(self, mocker, unity_server_address="localhost", unity_port_number=7000):
        self.UNITY_SERVER_URL = f"http://{unity_server_address}:{unity_port_number}"

        self.IS_UNITY_RUNNING = False

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

    def shutdown_server(self) -> None:
        os.kill(os.getpid(), signal.SIGTERM)

    @staticmethod
    def get_empty_base64() -> str:
        return base64.b64encode(cv2.imencode('.png', np.ones((100, 100, 3), dtype=np.uint8) * 255)[1]).decode('utf-8')

    def setup_calls(self) -> None:
        @self.app.get("/health")
        async def health_check():
            self.IS_UNITY_RUNNING = True
            return {"status": "OK"}

        @self.app.post("/capture_and_send")
        async def capture_and_send():
            if self.mocker.ret:
                frame = cv2.cvtColor(self.mocker.current_frame, cv2.COLOR_RGB2BGR)
                return image_to_base64(frame)

            return self.get_empty_base64()

        @self.app.post("/shutdown")
        async def shutdown():
            self.IS_UNITY_RUNNING = False
            self.shutdown_server()
            return {"message": "Mocker Capture Camera Server shutting down"}

        @self.app.on_event("shutdown")
        async def shutdown_event():
            if not self.IS_UNITY_RUNNING:
                async with httpx.AsyncClient() as client:
                    try:
                        await client.post(self.UNITY_SERVER_URL + "/shutdown/",
                                          json={"message": "Mocker Camera capture server is shutting down"})
                    except Exception as error:
                        logging.warning(f"Failed to send shutdown signal: {error}")
