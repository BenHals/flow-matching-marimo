from abc import abstractmethod
from typing import Protocol, override

import numpy as np
import numpy.typing as npt

from lib import domain


class VectorField(Protocol):
    def get_vector(
        self, location: domain.Location, timestep: float
    ) -> npt.NDArray[np.floating]: ...


class ConvergingVectorField(VectorField):
    def __init__(self, theta: float) -> None:
        self.theta: float = theta

    @override
    def get_vector(self, location: domain.Location, timestep: float) -> domain.Location:
        return -self.theta * location
