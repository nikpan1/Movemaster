import cv2
import requests
import numpy as np
import time
import threading
from flask import Flask, request

app = Flask(__name__)

UNITY_SERVER_URL = "http://localhost:5000/send_frame"
UNITY_SHUTDOWN_URL = "http://localhost:5000/shutdown"

is_running = True
unity_is_active = False

@app.route('/unity_shutdown', methods=['POST'])
def unity_shutdown():
    global unity_is_active
    print("Unity has stopped.")
    unity_is_active = False
    return '', 200

def start_flask_server():
    app.run(port=5001)

def check_unity_status():
    global unity_is_active
    while not unity_is_active:
        try:

            response = requests.get(UNITY_SERVER_URL, timeout=1)
            if response.status_code == 200:
                unity_is_active = True
                print("Unity is running, starting to send frames.")
        except requests.exceptions.RequestException:
            print("Waiting for Unity to start...")
            time.sleep(2)

def capture_and_send_frames():
    global is_running, unity_is_active
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open camera")
        return

    try:
        check_unity_status()

        while is_running and unity_is_active:
            ret, frame = cap.read()

            if not ret:
                print("Failed to capture frame")
                break

            _, encoded_image = cv2.imencode('.jpg', frame)
            image_data = encoded_image.tobytes()

            try:
                response = requests.post(UNITY_SERVER_URL, data=image_data, headers={'Content-Type': 'application/octet-stream'})
                if response.status_code == 200:
                    print("The frame sent")
                else:
                    print(f"Unity error: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error connecting to Unity: {e}")
                break

            time.sleep(0.1)

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if unity_is_active:
            try:
                requests.post(UNITY_SHUTDOWN_URL)
                print("Python is closed")
            except requests.exceptions.RequestException as e:
                print(f"Error connecting to Unity (shutdown): {e}")

if __name__ == "__main__":

    flask_thread = threading.Thread(target=start_flask_server)
    flask_thread.daemon = True  
    flask_thread.start()

    capture_and_send_frames()