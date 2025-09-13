from typing import Callable, Protocol, override

AlphaFn = Callable[[float], float]
BetaFn = Callable[[float], float]
MinVal = 1e-4


def alpha_basic(t: float):
    return max(t, MinVal)


def beta_basic(t: float):
    return max(1 - t, MinVal)


class AlphaBetaParam(Protocol):
    def get_value(self, t: float) -> float: ...

    def get_deriv(self, t: float) -> float: ...


class AlphaBasic(AlphaBetaParam):
    @override
    def get_value(self, t: float) -> float:
        return alpha_basic(t)

    @override
    def get_deriv(self, t: float) -> float:
        return 1


class BetaBasic(AlphaBetaParam):
    @override
    def get_value(self, t: float) -> float:
        return beta_basic(t)

    @override
    def get_deriv(self, t: float) -> float:
        return -1
