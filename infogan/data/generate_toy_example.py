import numpy as np


def generate_circle_toy_data():
    x1 = np.arange(-5, 5, 0.1)
    y1 = np.sqrt(5**2 - x1**2)

    x2 = np.arange(-5, 5, 0.1)
    y2 = -np.sqrt(5**2 - x2**2)

    x = np.concatenate([x1, x2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    return np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)