import numpy as np
import numpy.typing as npt

from lib import domain


class Trajectory:
    def __init__(
        self, locations: domain.LocationPath, timesteps: npt.NDArray[np.floating]
    ):
        self.locations = locations
        self.timesteps = timesteps
