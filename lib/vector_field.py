from abc import abstractmethod
from typing import Protocol, override

import numpy as np
import numpy.typing as npt

from lib import domain
from lib.alpha_beta import AlphaBetaParam, AlphaFn, BetaFn


class VectorField(Protocol):
    def get_vector(
        self, location: domain.Location, timestep: float
    ) -> npt.NDArray[np.floating]: ...


class GaussianConditionalVectorField(VectorField):
    def __init__(
        self, alpha: AlphaBetaParam, beta: AlphaBetaParam, z: domain.Location
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.z = z

    @override
    def get_vector(self, location: domain.Location, timestep: float) -> domain.Location:
        # print(f"""
        #       {self.alpha.get_deriv(timestep)}
        #     - ({self.beta.get_deriv(timestep)} / {self.beta.get_value(timestep)})
        #     * {self.alpha.get_value(timestep)}
        # ) * {self.z} + (
        #       {self.beta.get_deriv(timestep)} / {self.beta.get_value(timestep)}
        # ) * {location}
        #   """)
        return (
            self.alpha.get_deriv(timestep)
            - (self.beta.get_deriv(timestep) / self.beta.get_value(timestep))
            * self.alpha.get_value(timestep)
        ) * self.z + (
            self.beta.get_deriv(timestep) / self.beta.get_value(timestep)
        ) * location


class ConvergingVectorField(VectorField):
    def __init__(self, theta: float) -> None:
        self.theta: float = theta

    @override
    def get_vector(self, location: domain.Location, timestep: float) -> domain.Location:
        return -self.theta * location
