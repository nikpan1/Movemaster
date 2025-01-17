
import cv2
from datetime import datetime

class WebcamRecorder:
    def __init__(self) -> None:
        self.codec = cv2.VideoWriter_fourcc(*'XVID')
        self.output = None

    def arm_recording(self) -> None:
        current_date = datetime.now()
        formatted_date = current_date.strftime('%d%m%Y_%H%M%S')
        self.output = cv2.VideoWriter(formatted_date, self.codec, 20.0, (640, 480))

    def record_frame(self, frame: cv2.Mat) -> None:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        self.output.write(frame_hsv)

    def save_video(self) -> None:
        self.output.release()