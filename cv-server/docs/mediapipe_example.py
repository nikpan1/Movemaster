import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
given_point = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = mp_pose.process(frame)
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=4, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=6, circle_radius=2)
        )
        landmark = results.pose_landmarks.landmark[given_point]
        height, width, _ = frame.shape
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
    if cv2.waitKey(1) == ord('w'):
        given_point += 1
        if given_point > 32:
            given_point = 0

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()