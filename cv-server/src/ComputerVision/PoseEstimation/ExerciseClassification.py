from collections import Counter

from ComputerVision.PoseEstimation.PoseEstimation import *

from ComputerVision.PoseEstimation.model.conv_ae import ConvAutoencoder
from ComputerVision.PoseEstimation.model.multimodal_ae import MultimodalAutoencoder
from ComputerVision.PoseEstimation.model.multimodal_ar import MultimodalFcClassifier


class Args:
    def __init__(self):
        self.num_classes = 11
        self.window_length = 5
        self.window_stride = 0.2
        self.skeleton_sampling_rate = 30
        self.ae_layers = 3
        self.ae_hidden_units = 1000
        self.embedding_units = 1000
        self.ae_dropout = 0.0
        self.layers = 3
        self.hidden_units = 100
        self.dropout = 0.0


class ExerciseRecognition:
    def __init__(self, model_path, device, repetitiveness, args):
        self.model_path = model_path
        self.device = device
        self.repetitiveness = repetitiveness
        self.repetitiveness_counter = 0
        self.args = args

        self.pose_extractor = PoseLandmarkExtractor()

        self.labels_list = ['squats', 'lunges', 'bicep_curls', 'situps', 'pushups', 'tricep_extensions',
                            'dumbbell_rows', 'jumping_jacks', 'dumbbell_shoulder_press', 'lateral_shoulder_raises',
                            'non_activity']
        self.inference_history = deque(maxlen=10)
        self.non_activity_counter = 0

        self.predicted_class = 'non_activity'
        self.confidence = 0

        self.model = self.load_model()

    def load_model(self):
        skel_model = ConvAutoencoder(input_size=(self.args.window_length, 16), input_ch=2, dim=2, layers=3,
                                     grouped=[2, 2, 1], kernel_size=11, kernel_stride=(2, 1),
                                     return_embeddings=True).to(self.device)

        multimodal_ae_model = MultimodalAutoencoder(f_in=1620, skel=skel_model, layers=self.args.ae_layers,
                                                    hidden_units=self.args.ae_hidden_units,
                                                    f_embedding=self.args.embedding_units, dropout=self.args.ae_dropout,
                                                    return_embeddings=True).to(self.device)

        model = MultimodalFcClassifier(f_in=self.args.embedding_units, num_classes=self.args.num_classes,
                                       multimodal_ae_model=multimodal_ae_model, layers=self.args.layers,
                                       hidden_units=self.args.hidden_units, dropout=self.args.dropout).to(self.device)

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()

        return model

    def process_frame(self, landmarks_history):
        if self.pose_extractor.is_full():
            confidence, predicted_label = self.inference(landmarks_history)

            if predicted_label != 'non_activity' or confidence >= 0.66:
                self.inference_history.append((predicted_label, confidence))
                self.non_activity_counter = 0
                return self.get_weighted_most_common_element()

            else:
                self.non_activity_counter += 1

        if self.non_activity_counter >= 50 or not self.inference_history:
            self.inference_history = self.inference_history[
                                     -int(len(self.inference_history) / 2):] if self.inference_history else []
            self.non_activity_counter = 0
            return 'non_activity', 0

        return self.get_weighted_most_common_element()

    def inference(self, landmarks_history):
        input_data = PoseLandmarkExtractor.convert_history_to_tensor(landmarks_history, self.device)
        output = self.model(input_data)
        _, predicted_class_idx = torch.max(output, dim=1)
        predicted_label = self.labels_list[predicted_class_idx.item()]
        confidence = torch.softmax(output, dim=1)[0][predicted_class_idx].item()
        return confidence, predicted_label

    def get_weighted_most_common_element(self):
        weighted_counter = Counter()
        for label, confidence in self.inference_history:
            weighted_counter[label] += confidence

        most_common_element, total_confidence = weighted_counter.most_common(1)[0]
        return most_common_element, total_confidence / 100

    def recognize(self, frame):
        self.pose_extractor.extract_landmarks(frame)

        self.repetitiveness_counter += 1
        if self.repetitiveness_counter >= self.repetitiveness:
            self.repetitiveness_counter = 0

            landmarks_history = self.pose_extractor.get_landmarks_history()
            self.predicted_class, self.confidence = self.process_frame(landmarks_history)

        return self.predicted_class, self.confidence


if __name__ == '__main__':
    args = Args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    exercise_recognition = ExerciseRecognition(model_path=r"model.pth", device=device, args=args)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        predicted_class, confidence = exercise_recognition.recognize(frame)

        print(predicted_class)
        cv2.imshow('Pose Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
