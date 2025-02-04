from PoseEstimation import *
from mmfitexample import args, instantiate_model
from my_utils.data_transforms import Unit

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = ("C:/Users/nikod/Desktop/movemaster-nn/ExerciseClassification/"
              "output-optimized/best_model.pth")
video_path = 1


class ExerciseRecognition:
    def __init__(self, model_path, device, args):
        self.model_path = model_path
        self.device = device
        self.args = args
        self.labels_list = args.ACTIONS

        self.pose_extractor = PoseLandmarkExtractor()
        self.model = self.load_model()

    def load_model(self):
        model = instantiate_model()

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()

        return model

    def process_frame(self, landmarks_history):
        if self.pose_extractor.is_full():
            confidence, predicted_label = self.inference(landmarks_history)
            return predicted_label, confidence
        return 'non_activity', 0

    def inference(self, landmarks_history):
        input_data = PoseLandmarkExtractor.convert_history_to_tensor(landmarks_history, self.device)

        # Ensure input_data has the correct shape for Unit processing
        input_data_np = input_data.cpu().numpy()
        unit_transform = Unit()
        processed_data_np = unit_transform(input_data_np[0])

        processed_data_tensor = torch.from_numpy(processed_data_np).unsqueeze(0).to(self.device, dtype=input_data.dtype)
        input_data = processed_data_tensor

        # Pass through the model
        output = self.model(input_data)
        _, predicted_class_idx = torch.max(output, dim=1)
        predicted_label = self.labels_list[predicted_class_idx.item()]
        confidence = torch.softmax(output, dim=1)[0][predicted_class_idx].item()
        return confidence, predicted_label

    def predict(self, frame):
        self.pose_extractor.extract_landmarks(frame)
        landmarks_history = self.pose_extractor.get_landmarks_history()
        predicted_class, confidence = self.process_frame(landmarks_history)
        return predicted_class, confidence


if __name__ == '__main__':
    exercise_recognition = ExerciseRecognition(model_path=model_path, device=device, args=args)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}.")
        exit(1)

    isEvenFrame = True
    predicted_class, confidence = "a", 0.0

    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, [int(frame.shape[1] / 4), int(frame.shape[0] / 4)])
        if not ret or frame is None:
            print("Error: Failed to read frame or end of video reached.")
            break

        # Predict every alternate frame
        predicted_class, confidence = exercise_recognition.predict(frame)
        print(predicted_class, confidence)

        # Add text BEFORE showing the frame
        cv2.putText(frame, f"{predicted_class}, {confidence*100:.2f}%",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # Now display the frame with the text
        cv2.imshow('Pose Estimation', frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()