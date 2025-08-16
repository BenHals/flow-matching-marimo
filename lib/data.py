import math

import numpy as np
import numpy.typing as npt

from lib.domain import Dataset, Location


def make_spiral(n_spiral_arms: int, n_points: int) -> Dataset:
    ts = np.random.uniform(low=0, high=1, size=n_points)
    x = [math.sin(t * n_spiral_arms * (2 * math.pi)) * t for t in ts]
    y = [math.cos(t * n_spiral_arms * (2 * math.pi)) * t for t in ts]

    return [np.array([xv, yv]) for xv, yv in zip(x, y)]


def make_gaussian(n_points: int):
    data = np.random.standard_normal((n_points, 2))
    return list(data[:])


def sample_dataset(dataset: Dataset) -> Location:
    return dataset[0]
