from typing import Callable

AlphaFn = Callable[[float], float]
BetaFn = Callable[[float], float]


def alpha_basic(t: float):
    return 1 - t


def beta_basic(t: float):
    return t
