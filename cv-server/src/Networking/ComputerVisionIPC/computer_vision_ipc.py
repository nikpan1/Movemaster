import json
import time
import zmq
import cv2
import numpy as np
from ComputerVision.PoseEstimation.PoseEstimation import PoseLandmarkExtractor
import base64


class ComputerVisionIpcServer:
    def __init__(self, host="localhost", port=5556):
        # Initialize ZMQ context and server
        self.context = zmq.Context()
        self.server = self.context.socket(zmq.REP)
        self.server.bind(f"tcp://{host}:{port}")

        # Initialize PoseLandmarkExtractor
        self.lm = PoseLandmarkExtractor()

    def base64_to_image(self, input_str: str) -> cv2.Mat:
        '''
        Function to convert base64 string to image file
        '''
        try:
            # Clean base64 string (remove prefix if exists)
            if ',' in input_str:
                input_str = input_str.split(',')[1]

            # Strip any extra whitespace
            input_str = input_str.strip()

            # Decode the base64 string to bytes
            img_data = base64.b64decode(input_str)

            # Check if the data length is reasonable
            print(f"Decoded base64 data length: {len(img_data)} bytes")  # Debugging length
            if len(img_data) < 100:  # If the image size is unusually small, there's an issue
                print(f"Warning: Decoded image size is too small ({len(img_data)} bytes).")
                return None

            # Convert the byte data into an image array using OpenCV
            np_array = np.frombuffer(img_data, np.uint8)
            image_mat = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            # Check if the image was decoded successfully
            if image_mat is None:
                print("Failed to decode image with OpenCV.")
                return None

            return image_mat

        except Exception as e:
            print(f"Error during base64-to-image conversion: {e}")
            return None

    def start_server(self):
        '''Method to start the server and handle incoming requests'''
        while True:
            try:
                # Receive the message containing the base64 image
                message = self.server.recv_string()

                # Parse the received message
                data = json.loads(message)
                base64_image = data.get("base64_image", "")

                # Print first 10 characters and last 10 characters of the base64 image
                if base64_image:
                    print(f"First 10 characters: {base64_image[:10]}")
                    print(f"Last 10 characters: {base64_image[-10:]}")

                if not base64_image:
                    print("No image received, retrying...")
                    time.sleep(20)
                    continue  # Skip processing and wait for new data

                # Decode the base64 image
                image = self.base64_to_image(base64_image)

                if image is None:
                    print("Failed to decode base64 image.")
                    continue  # Skip invalid image processing

                # Extract pose landmarks (33x3 coordinates)
                result = self.lm.extract_landmarks(image)
                if result is None:
                    print("Pose extraction failed.")
                    continue
                print(result)

                # Pack the result into a JSON object with the key 'points'
                result_packed = json.dumps({"points": result.tolist()})
                self.server.send_string(result_packed)

            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(20)  # Wait before retrying