import cv2
import mediapipe as mp
import numpy as np


class PoseLandmarkExtractor:
    def __init__(self, min_detection_confidence: float = 0.7, min_tracking_confidence: float = 0.7):
        """
        Initializes the pose processor with MediaPipe Pose.

        :param min_detection_confidence: Minimum confidence value for pose detection.
        :param min_tracking_confidence: Minimum confidence value for pose tracking.
        """
        self.mp_pose = mp.solutions.pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_landmarks(self, frame: cv2.Mat) -> np.ndarray:
        """
        Process the given frame to detect pose landmarks and return a fixed-size array of (33 landmarks * 3 values).

        Each landmark will have three values: [x, y, visibility].
        If a landmark is detected, x and y will hold the coordinates, and visibility will be 1.0.
        If not detected, x and y will be 0, and visibility will be 0.0.

        :param frame: Input frame in the form of a cv2.Mat.
        :return: A NumPy array of shape (33, 3) where each row represents [x, y, visibility].
        """
        landmarks_array = np.zeros((33, 3), dtype=np.float32)

        results = self.mp_pose.process(frame)

        if results.pose_landmarks is not None:
            height, width, _ = frame.shape

            for i, landmark in enumerate(results.pose_landmarks.landmark):
                if i >= 33:
                    break

                landmarks_array[i, 0] = landmark.x * width
                landmarks_array[i, 1] = landmark.y * height
                landmarks_array[i, 2] = 1.0

        return landmarks_array

    def draw_landmarks(self, image: cv2.Mat, landmarks: np.ndarray):
        """
        Draws the pose landmarks on the image based on the extracted landmarks.

        :param image: Input image in the form of a cv2.Mat.
        :param landmarks: NumPy array of shape (33, 3), representing the pose landmarks.
        """
        for landmark in landmarks:
            x, y, visibility = landmark

            if visibility == 1.0:
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        return image


def place_on_canvas(image: cv2.Mat) -> cv2.Mat:
    """
    Places the input image in the center of a 2x larger canvas.

    :param image: Input image in the form of a cv2.Mat.
    :return: The larger canvas with the image placed at the center.
    """
    height, width, _ = image.shape

    # Create a 2x larger empty canvas (black background)
    canvas = np.zeros((height * 3, width * 3, 3), dtype=np.uint8)

    # Calculate the position to place the image at the center
    y_offset = height // 2
    x_offset = width // 2

    # Place the input image at the center of the canvas
    canvas[y_offset:y_offset + height, x_offset:x_offset + width] = image

    return canvas
