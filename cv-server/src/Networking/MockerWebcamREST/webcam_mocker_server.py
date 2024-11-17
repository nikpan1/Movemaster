import json
import logging
import os
import signal
import threading
import httpx
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from fastapi import FastAPI
import numpy as np
from Networking.Base64.base64_conversions import *

class MockerCaptureCameraServer:
    def __init__(self, mocker, unity_server_address="localhost", unity_port_number=7000, max_threads=10):
        self.UNITY_SERVER_URL = f"http://{unity_server_address}:{unity_port_number}"
        self.IS_UNITY_RUNNING = False
        self.current_frame = self.get_empty_image()
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
    def get_empty_image() -> np.ndarray:
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

        @self.app.get("/capture_and_send")
        async def capture_and_send():
            frame = self.mocker.current_frame
            image_base64 = image_to_base64(frame)

            return image_base64

        @self.app.on_event("shutdown")
        async def shutdown_event():
            if not self.IS_UNITY_RUNNING:
                async with httpx.AsyncClient() as client:
                    try:
                        await client.post(self.UNITY_SERVER_URL + "/shutdown/",
                                          json={"message": "Mocker Camera capture server is shutting down"})
                    except Exception as error:
                        logging.warning(f"Failed to send shutdown signal: {error}")

    def send_frame_to_unity_sync(self, input_string):
        """
        Sends a synchronous PUT request with a string payload to the Unity server.
        """
        headers = {
            "Content-Type": "application/json"
        }

        # Try to acquire the semaphore without blocking
        if self.semaphore.acquire(blocking=False):
            with httpx.Client() as client:
                try:
                    response = client.put(self.UNITY_SERVER_URL + '/new_frame', data='{"key": "value"}', headers=headers)
                    response.raise_for_status()
                    print(f"String sent successfully: {response.status_code}")
                    return response.status_code
                except httpx.RequestError as e:
                    print(f"Error sending string to Unity server: {e}")
                    return None
                finally:
                    # Release the semaphore after the request is done
                    self.semaphore.release()
        else:
            # If semaphore could not be acquired, discard the request
            print("Semaphore limit reached. Request discarded.")
            return None

    def send_async_frame_to_unity(self, input_string):
        """
        Executes the send_frame_to_unity_sync method asynchronously, managing it through a thread pool.
        """
        # Submit the task to the thread pool
        self.executor.submit(self.send_frame_to_unity_sync, input_string)
