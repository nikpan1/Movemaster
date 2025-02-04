from PoseEstimation import *

cap = cv2.VideoCapture(r"C:\Users\nikod\Downloads\VID_20250113_205838.mp4")
pose_extractor = PoseLandmarkExtractor()

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    sizes = frame.shape
    scale = [int(sizes[1] / 8), int(sizes[0] / 8)]
    frame = cv2.resize(frame, scale)
    if not ret:
        print("Error: Failed to capture frame.")
        break

    landmarks = pose_extractor.extract_landmarks(frame)

    frame_with_landmarks = pose_extractor.draw_landmarks(frame, landmarks)

    if pose_extractor.is_full():
        print("Is full!")

    # Display the frame
    cv2.imshow('Pose Estimation', frame_with_landmarks)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
