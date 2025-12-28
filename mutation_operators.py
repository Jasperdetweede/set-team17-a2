import numpy as np
from scipy.ndimage import gaussian_filter


def generate_additive_noise_neighbour(seed: np.ndarray, epsilon: float) -> np.ndarray:
    neighbour = seed.copy()
    h, w, c = seed.shape
    limit = 255 * epsilon
    noise = np.random.uniform(-limit, limit, (h, w, c))
    neighbour = np.clip(neighbour + noise, 0, 255)
    return neighbour

def generate_local_masking_neighbour(seed: np.ndarray, epsilon: float) -> np.ndarray:
    neighbour = gaussian_filter(seed, sigma=(1.0, 1.0, 0.0))
    neighbour = neighbour.astype(np.float32)

    limit = 255 * epsilon
    mask = np.abs(neighbour - seed) < limit
    neighbour = np.where(mask, neighbour, seed)
    neighbour = np.clip(neighbour, 0, 255)
    return neighbour

def channel_specific_perturbation_neighbour(seed: np.ndarray, epsilon: float) -> np.ndarray:
    neighbour = seed.copy()
    h, w, c = seed.shape
    limit = 255 * epsilon
    for channel in range(c):
        noise = np.random.uniform(-limit, limit, (h, w))
        neighbour[:, :, channel] = np.clip(neighbour[:, :, channel] + noise, 0, 255)
    return neighbour

def generate_lines_neighbour(seed: np.ndarray, epsilon: float, num_lines: int = 3, width: int = 3) -> np.ndarray:
    neighbour = seed.copy()
    h, w, c = seed.shape
    limit = 255 * epsilon

    X, Y = np.meshgrid(np.arange(w), np.arange(h))

    for _ in range(num_lines):
        slope = np.random.uniform(-5.0, 5.0)
        intercept = np.random.uniform(0, h) - slope * (0.5 * w) 

        # compute distance from line y = slope * x + intercept
        dist = np.abs(Y - (slope * X + intercept))
        mask = dist <= width  # boolean mask where line affects

        # generate random perturbation for masked pixels
        delta = np.random.uniform(-limit, limit, size=(h, w, c))
        for ch in range(c):
            clipped  = np.clip(neighbour[:, :, ch] + delta[:, :, ch], 0, 255)
            neighbour[:, :, ch] = np.where(mask, clipped, neighbour[:, :, ch])

    return neighbour