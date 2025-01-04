import cv2
import mediapipe as mp
import numpy as np
from collections import deque

import torch


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

        self.landmark_history = deque(maxlen=90)

    def get_landmarks_history(self):
        return self.landmark_history

    def is_full(self):
        return len(self.landmark_history) == self.landmark_history.maxlen

    @staticmethod
    def convert_history_to_tensor(landmark_history: deque, device) -> torch.Tensor:
        """
        Converts the landmark history into a PyTorch tensor of shape (1, 2, 150, 18).

        :param landmark_history: A deque containing the last 150 frames of landmarks, each as a NumPy array (18, 2).
        :return: A PyTorch tensor of shape (1, 2, 150, 18).
        """

        # Stack the history into a NumPy array of shape (150, 18, 2)
        history_array = np.stack(landmark_history, axis=0)  # Shape: (150, 18, 2)

        # Permute and reshape into (1, 2, 150, 18) as required
        tensor = torch.from_numpy(history_array).permute(2, 0, 1).unsqueeze(0).float().to(device)  # Shape: (1, 2, 150, 18)

        return tensor

    def extract_landmarks(self, frame: cv2.Mat) -> np.ndarray:
        """
        Process the given frame to detect pose landmarks and return a fixed-size array of (33 landmarks * 3 values),
        where each landmark contains [x, y, z] coordinates.

        :param frame: Input frame in the form of a cv2.Mat.
        :return: A NumPy array of shape (33, 3) where each row represents [x, y, z].
        """
        xyz = np.zeros((33, 3), dtype=np.float32)

        results = self.mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks is not None:
            height, width, _ = frame.shape

            for i, landmark in enumerate(results.pose_landmarks.landmark):
                if i >= 33:
                    break

                xyz[i, 0] = landmark.x * width  # x-coordinate
                xyz[i, 1] = landmark.y * height  # y-coordinate
                xyz[i, 2] = landmark.z  # z-coordinate

        landmarks = self.mediapipe_to_openpose(xyz)
        self.landmark_history.append(landmarks)

        return landmarks

    def draw_landmarks(self, image: cv2.Mat, landmarks: np.ndarray):
        """
        Draws the pose landmarks on the image based on the extracted landmarks.

        :param image: Input image in the form of a cv2.Mat.
        :param landmarks: NumPy array of shape (33, 3), representing the pose landmarks.
        """
        for landmark in landmarks:
            x, y = landmark

            if x > 0 and y > 0:
                cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

        return image

    def mediapipe_to_openpose(self, mediapipe_landmarks: np.ndarray) -> np.ndarray:
        """
        Converts MediaPipe pose landmarks (33 keypoints) to OpenPose format (18 keypoints).

        :param mediapipe_landmarks: NumPy array of shape (33, 3) representing MediaPipe landmarks.
        :return: NumPy array of shape (18, 2) representing OpenPose landmarks.
        """
        openpose_landmarks = np.zeros((18, 2), dtype=np.float32)

        def avg_points(p1, p2):
            return (mediapipe_landmarks[p1, :2] + mediapipe_landmarks[p2, :2]) / 2

        mapping = {
            0: 0,
            1: (12, 11),
            2: 12,
            3: 14,
            4: 16,
            5: 11,
            6: 13,
            7: 15,
            8: 24,
            9: 26,
            10: 28,
            11: 23,
            12: 25,
            13: 27,
            14: 5,
            15: 2,
            16: 8,
            17: 7
        }

        for openpose_idx, mediapipe_key in mapping.items():
            if isinstance(mediapipe_key, tuple):
                openpose_landmarks[openpose_idx] = avg_points(*mediapipe_key)
            else:
                openpose_landmarks[openpose_idx] = mediapipe_landmarks[mediapipe_key, :2]

        return openpose_landmarks
