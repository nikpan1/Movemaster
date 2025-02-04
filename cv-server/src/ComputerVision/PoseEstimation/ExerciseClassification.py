from collections import Counter

from ComputerVision.PoseEstimation.PoseEstimation import *

from ComputerVision.PoseEstimation.model.conv_ae import ConvAutoencoder
from ComputerVision.PoseEstimation.model.multimodal_ae import MultimodalAutoencoder
from ComputerVision.PoseEstimation.model.multimodal_ar import MultimodalFcClassifier


class Unit(object):
    """Transforms the input sample and appends calculated values to expand dimensions."""

    def __call__(self, sample):
        additional_values = np.ones((sample.shape[0], sample.shape[1], 31 - sample.shape[2]))

        for i in range(sample.shape[1]):  # Iterate over the second dimension
            additional_values = Unit.calc_vec(additional_values, sample, 3, 4, 0, i)
            additional_values = Unit.calc_vec(additional_values, sample, 2, 3, 1, i)
            additional_values = Unit.calc_vec(additional_values, sample, 1, 2, 2, i)
            additional_values = Unit.calc_vec(additional_values, sample, 5, 1, 3, i)
            additional_values = Unit.calc_vec(additional_values, sample, 6, 5, 4, i)
            additional_values = Unit.calc_vec(additional_values, sample, 7, 6, 5, i)
            additional_values = Unit.calc_vec(additional_values, sample, 1, 8, 6, i)
            additional_values = Unit.calc_vec(additional_values, sample, 1, 11, 7, i)
            additional_values = Unit.calc_vec(additional_values, sample, 8, 9, 8, i)
            additional_values = Unit.calc_vec(additional_values, sample, 10, 8, 9, i)
            additional_values = Unit.calc_vec(additional_values, sample, 11, 12, 10, i)
            additional_values = Unit.calc_vec(additional_values, sample, 12, 13, 11, i)
            additional_values = Unit.calc_vec(additional_values, sample, 0, 1, 12, i)

        return np.concatenate((sample, additional_values), axis=2)

    @staticmethod
    def calc_vec(additional_values, sample, f, s, r, i):
        additional_values[0, i, r] = abs(sample[0, i, f] - sample[0, i, s])
        additional_values[1, i, r] = abs(sample[1, i, f] - sample[1, i, s])
        return additional_values

class Args:
    def __init__(self):
        self.num_classes = 11
        self.window_length = 3
        self.window_stride = 0.2
        self.skeleton_sampling_rate = 30
        self.epochs = 150
        self.ae_layers = 5
        self.ae_hidden_units = 2048
        self.embedding_units = 2048
        self.ae_dropout = 0.1
        self.layers = 5
        self.hidden_units = 256
        self.dropout = 0.1

        self.lr = 1e-3
        self.batch_size = 128
        self.eval_every = 1
        self.early_stop = 15
        self.checkpoint = 10


class ExerciseRecognition:
    def __init__(self, model_path):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args = Args()
        self.pose_extractor = PoseLandmarkExtractor()
        self.labels_list = ['squats', 'lunges', 'bicep_curls', 'situps', 'pushups', 'tricep_extensions',
                            'dumbbell_rows', 'jumping_jacks', 'dumbbell_shoulder_press', 'lateral_shoulder_raises',
                            'non_activity']
        self.non_activity_counter = 0
        self.model = self.load_model(model_path)

        self.predicted_class = 'non_activity'
        self.confidence = 0

    def load_model(self, model_path):
        # Precompute skeleton window length
        skeleton_window_length = int(self.args.window_length * self.args.skeleton_sampling_rate)

        # Initialize skeleton model
        skel_model = ConvAutoencoder(
            input_size=(skeleton_window_length, 16),
            input_ch=2,
            dim=2,
            layers=3,
            grouped=[2, 2, 2],
            kernel_size=5,
            kernel_stride=(1, 1),
            return_embeddings=True
        ).to(self.device, non_blocking=True)

        # Initialize multimodal autoencoder
        multimodal_ae_model = MultimodalAutoencoder(
            f_in=16200,
            layers=5,
            dropout=0.3,
            hidden_units=128,
            f_embedding=256,
            skel=skel_model,
            return_embeddings=False
        ).to(self.device, non_blocking=True)

        # Initialize the classifier
        model = MultimodalFcClassifier(
            f_in=2880,
            num_classes=self.args.num_classes,
            multimodal_ae_model=multimodal_ae_model,
            layers=5,
            hidden_units=128,
            dropout=0.3
        ).to(self.device, non_blocking=True)

        # Load and map checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Set to evaluation mode
        model.eval()

        return model

    def process_frame(self, landmarks_history):
        if self.pose_extractor.is_full():
            self.confidence, self.predicted_class = self.inference(landmarks_history)

        return self.predicted_class, self.confidence

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
        conf = torch.softmax(output, dim=1)[0][predicted_class_idx].item()
        print(predicted_label)
        return conf, predicted_label

    def recognize(self, frame):
        self.pose_extractor.extract_landmarks(frame)
        landmarks_history = self.pose_extractor.get_landmarks_history()
        predicted_label, conf = self.process_frame(landmarks_history)
        return predicted_label, conf


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
