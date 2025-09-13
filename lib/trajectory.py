import numpy as np
import numpy.typing as npt

from lib import domain


class Trajectory:
    """A trajectory is a function that takes a timestep [0, 1] as input,
    and outputs a location in R^D. The intuition is that the trajectory is
    describing the movement of a point over time.
    """

    def __init__(
        self, locations: domain.LocationPath, timesteps: npt.NDArray[np.floating]
    ):
        self.locations = locations
        self.timesteps = timesteps
