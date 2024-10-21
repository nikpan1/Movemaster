from src.PoseEstimation.PoseEstimation import *

IMAGE_FILENAME = "PoseEstimation/person.jpg"

def test_is_working_correctly_PoseLandmarkExtractor():
    # GIVEN:
    image_path = IMAGE_FILENAME
    image = cv2.imread(image_path)

    # Check if the image is loaded correctly
    if image is None:
        raise ValueError(f"Error loading image from {image_path}")

    # Resize and add to canvas
    image = cv2.resize(image, (int(image.shape[1] / 2.5), int(image.shape[0] / 2.5)), interpolation=cv2.INTER_AREA)
    image = place_on_canvas(image)

    # WHEN: Initialize the pose landmark extractor
    pl = PoseLandmarkExtractor()
    landmarks = pl.extract_landmarks(image)

    # THEN:
    # Check if the landmarks have the correct shape
    assert landmarks.shape == (33, 3), f"Landmarks shape is incorrect: {landmarks.shape}"

    # Check if landmarks contain valid visibility flags (0.0 or 1.0)
    assert np.all(np.logical_or(landmarks[:, 2] == 0.0, landmarks[:, 2] == 1.0)), "Invalid visibility values in landmarks"

    # Draw landmarks on the image
    image_with_landmarks = pl.draw_landmarks(image, landmarks)

    # Check if the image is not empty after drawing
    assert image_with_landmarks is not None and np.any(image_with_landmarks != 0), "Image is empty after drawing landmarks"