import numpy as np

class Unit:
    """Transforms the input sample by normalizing and appending calculated values to expand dimensions."""

    def __call__(self, sample):
        sample = Unit.normalize(sample)  # Normalize before adding new features
        return Unit.unit(sample)

    @staticmethod
    def normalize(sample):
        """Normalizes the input sample by centering and scaling."""
        """Normalizes input sample using Min-Max Scaling to the range [-1, 1]."""
        min_val = np.min(sample, axis=2, keepdims=True)
        max_val = np.max(sample, axis=2, keepdims=True)
        return 2 * (sample - min_val) / (max_val - min_val + 1e-8) - 1  # Scale to [-1,1]

    @staticmethod
    def unit(sample):
        """Expands the sample by computing additional feature values."""
        additional_values = np.ones((sample.shape[0], sample.shape[1], 31 - sample.shape[2]))
        Unit.compute_additional_values(additional_values, sample)
        return np.concatenate((sample, additional_values), axis=2)

    @staticmethod
    def calc_vec(additional_values, sample, f, s, r, i):
        """Calculates the absolute difference between two feature indices and stores the result."""
        additional_values[:, i, r] = np.abs(sample[:, i, f] - sample[:, i, s])

    @staticmethod
    def compute_additional_values(additional_values, sample):
        """Populates additional_values with computed vectors based on joint distances."""
        calculations = [
            (3, 4, 0), (2, 3, 1), (1, 2, 2), (5, 1, 3), (6, 5, 4),
            (7, 6, 5), (1, 8, 6), (1, 11, 7), (8, 9, 8), (10, 8, 9),
            (11, 12, 10), (12, 13, 11), (0, 1, 12)
        ]

        for i in range(sample.shape[1]):  # Iterate over frames
            for f, s, r in calculations:
                Unit.calc_vec(additional_values, sample, f, s, r, i)


class Jitter:
    """Applies small random noise to pose data to improve robustness and appends additional values."""

    def __init__(self, sigma=0.05):
        self.sigma = sigma  # Controls noise magnitude

    def __call__(self, sample):
        sample = Unit.normalize(sample)
        sample = sample + np.random.normal(0, self.sigma, sample.shape)
        return Unit().unit(sample)


class HorizontalFlip:
    """Flips pose horizontally with a probability of 50% and appends additional values."""

    def __call__(self, sample):
        sample = Unit.normalize(sample)
        sample[:, :, 0] *= -1
        return Unit().unit(sample)


class GaussianNoise:
    """Adds random Gaussian noise to pose data to improve model robustness."""
    def __init__(self, mean=0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample = Unit.normalize(sample)

        noise = np.random.normal(self.mean, self.std, sample.shape)

        return Unit().unit(sample + noise)
