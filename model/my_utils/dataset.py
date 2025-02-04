import torch
from torch.utils.data import Dataset, Sampler

import my_utils.utils as utils


class MMFit(Dataset):
    """
    MM-Fit PyTorch Dataset class.
    """

    def __init__(self, modality_filepaths, label_path, window_length, skeleton_window_length, skeleton_transform):
        """
        Initialize MMFit Dataset object.
        :param modality_filepaths: Modality - file path mapping (dict) for a workout.
        :param label_path: File path to MM-Fit CSV label file for a workout.
        :param window_length: Window length in seconds.
        :param skeleton_window_length: Skeleton window length in number of samples.
        :param sensor_window_length: Sensor window length in number of samples.
        :param skeleton_transform: Transformation functions to apply to skeleton data.
        :param sensor_transform: Transformation functions to apply to sensor data.
        """
        self.window_length = window_length
        self.skeleton_window_length = skeleton_window_length
        self.skeleton_transform = skeleton_transform
        self.modalities = {}
        for modality, filepath in modality_filepaths.items():
            self.modalities[modality] = utils.load_modality(filepath)

        self.ACTIONS = {'squats': 0, 'lunges': 1, 'bicep_curls': 2, 'situps': 3, 'pushups': 4, 'tricep_extensions': 5,
                        'dumbbell_rows': 6, 'jumping_jacks': 7, 'dumbbell_shoulder_press': 8,
                        'lateral_shoulder_raises': 9, 'non_activity': 10}
        self.labels = utils.load_labels(label_path)

    def __len__(self):
        return self.modalities['pose_2d'].shape[1] - self.skeleton_window_length - 30

    def __getitem__(self, i):
        frame = self.modalities['pose_2d'][0, i, 0]
        sample_modalities = {}
        label = 'non_activity'
        reps = 0
        for row in self.labels:
            if (frame > (row[0] - self.skeleton_window_length / 2)) and (
                    frame < (row[1] - self.skeleton_window_length / 2)):
                label = row[3]
                reps = row[2]
                break

        for modality, data in self.modalities.items():
            if data is None:
                sample_modalities[modality] = torch.zeros(2, self.skeleton_window_length, 17)
            else:
                if 'pose' in modality:
                    sample_modalities[modality] = torch.as_tensor(self.skeleton_transform(
                        data[:, i:i + self.skeleton_window_length, 1:]), dtype=torch.float)

        return sample_modalities, self.ACTIONS[label], reps


class SequentialStridedSampler(Sampler):
    """
    PyTorch Sampler Class to sample elements sequentially using a specified stride, always in the same order.
    Arguments:
        data_source (Dataset):
        stride (int):
    """

    def __init__(self, data_source, stride):
        """
        Initialize SequentialStridedSampler object.
        :param data_source: Dataset to sample from.
        :param stride: Stride to slide window in seconds.
        """
        self.data_source = data_source
        self.stride = stride

    def __len__(self):
        return len(range(0, len(self.data_source), self.stride))

    def __iter__(self):
        return iter(range(0, len(self.data_source), self.stride))