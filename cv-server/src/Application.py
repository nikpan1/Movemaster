import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

from Networking.ComputerVisionIPC.computer_vision_ipc import ComputerVisionIpcServer
from Networking.MockerWebcamREST.webcam_mocker_application import WebcamMockerApplication
from Networking.WebcamREST.webcam_capture_server import WebcamCaptureServer

# Config for logging settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main(mock_mode):
    with ThreadPoolExecutor() as executor:
        # Start the computer vision server asynchronously
        logging.info("Running computer vision server.")
        computer_vision_server = ComputerVisionIpcServer()
        executor.submit(computer_vision_server.start_server)

        # Start either the mock or the real webcam application based on the flag
        if mock_mode:
            logging.info("Running webcam mocker.")
            webcam_mock_application = WebcamMockerApplication()
            webcam_mock_application.start_application()
        else:
            logging.info("Running webcam image capture server.")
            webcam_image_server = WebcamCaptureServer()
            executor.submit(webcam_image_server.start_server)


if __name__ == "__main__":
    # Setup for --mock flag handling
    parser = argparse.ArgumentParser(description="Run either mock or real webcam application.")
    parser.add_argument("--mock", action="store_true", help="Run the WebcamMockerApplication if this flag is set.")
    args = parser.parse_args()

    main(mock_mode=args.mock)
