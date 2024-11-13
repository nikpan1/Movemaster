import argparse

from Networking.ComputerVisionREST.computer_vision_server import ComputerVisionServer
from Networking.MockerWebcamREST.webcam_mocker_application import WebcamMockerApplication
from Networking.WebcamREST.webcam_capture_server import WebcamCaptureServer


def main(mock_mode):
    # Always start the computer vision server
    computer_vision_server = ComputerVisionServer()
    computer_vision_server.start_server()

    # Start either the mock or the real webcam application based on the flag
    if mock_mode:
        webcam_mock_application = WebcamMockerApplication()
        webcam_mock_application.start_application()
    else:
        webcam_image_server = WebcamCaptureServer()
        webcam_image_server.start_server()


if __name__ == "__main__":
    # Setup for --mock flag handling
    parser = argparse.ArgumentParser(description="Run either mock or real webcam application.")
    parser.add_argument("--mock", action="store_true", help="Run the WebcamMockerApplication if this flag is set.")
    args = parser.parse_args()

    main(mock_mode=args.mock)
