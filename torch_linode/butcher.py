from dataclasses import dataclass
import torch
import math

@dataclass
class ButcherTableau:
    """Represents the coefficients of a Runge-Kutta method."""
    c: torch.Tensor
    b: torch.Tensor
    a: torch.Tensor
    order: int
    b_error: torch.Tensor = None

DOPRI5 = ButcherTableau(
    a=torch.tensor([
        [0, 0, 0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
    ], dtype=torch.float64),
    b=torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0], dtype=torch.float64),
    c=torch.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1], dtype=torch.float64),
    b_error=torch.tensor([
        35 / 384 - 1951 / 22680, 0, 500 / 1113 - 451 / 720, 125 / 192 - 51 / 160,
        -2187 / 6784 - 22075 / 100000, 11 / 84 - 1 / 40, 0
    ], dtype=torch.float64),
    order=5
)


RK4 = ButcherTableau(
    a=torch.tensor([
        [0, 0, 0, 0],
        [1 / 2, 0, 0, 0],
        [0, 1 / 2, 0, 0],
        [0, 0, 1, 0],
    ], dtype=torch.float64),
    b=torch.tensor([1 / 6, 1 / 3, 1 / 3, 1 / 6], dtype=torch.float64),
    c=torch.tensor([0, 1 / 2, 1 / 2, 1], dtype=torch.float64),
    order=4
)


GL2 = ButcherTableau(
    a=torch.tensor([
        [1 / 4, 1 / 4 - math.sqrt(3) / 6],
        [1 / 4 + math.sqrt(3) / 6, 1 / 4],
    ], dtype=torch.float64),
    b=torch.tensor([1 / 2, 1 / 2], dtype=torch.float64),
    c=torch.tensor([1 / 2 - math.sqrt(3) / 6, 1 / 2 + math.sqrt(3) / 6], dtype=torch.float64),
    order=4
)

GL4 = ButcherTableau(
    a=torch.tensor([
        [1/4, 1/4 - math.sqrt(3) / 6],
        [1/4 + math.sqrt(3) / 6, 1/4]
    ], dtype=torch.float64),
    b=torch.tensor([1/2, 1/2], dtype=torch.float64),
    c=torch.tensor([1/2 - math.sqrt(3) / 6, 1/2 + math.sqrt(3) / 6], dtype=torch.float64),
    order=4
)

GL6 = ButcherTableau(
    a=torch.tensor([
        [5 / 36, 2 / 9 - math.sqrt(15) / 15, 5 / 36 - math.sqrt(15) / 30],
        [5 / 36 + math.sqrt(15) / 24, 2 / 9, 5 / 36 - math.sqrt(15) / 24],
        [5 / 36 + math.sqrt(15) / 30, 2 / 9 + math.sqrt(15) / 15, 5 / 36],
    ], dtype=torch.float64),
    b=torch.tensor([5 / 18, 4 / 9, 5 / 18], dtype=torch.float64),
    c=torch.tensor([1 / 2 - math.sqrt(15) / 10, 1 / 2, 1 / 2 + math.sqrt(15) / 10], dtype=torch.float64),
    order=6
)


RADAU2 = ButcherTableau(
    a=torch.tensor([[1]], dtype=torch.float64),
    b=torch.tensor([1], dtype=torch.float64),
    c=torch.tensor([1], dtype=torch.float64),
    order=1
)


RADAU4 = ButcherTableau(
    a=torch.tensor([
        [5 / 12, -1 / 12],
        [3 / 4, 1 / 4],
    ], dtype=torch.float64),
    b=torch.tensor([3 / 4, 1 / 4], dtype=torch.float64),
    c=torch.tensor([1 / 3, 1], dtype=torch.float64),
    order=3
)


RADAU6 = ButcherTableau(
    a=torch.tensor([
        [(88 - 7 * math.sqrt(6)) / 360, (296 - 169 * math.sqrt(6)) / 1800, (-2 + 3 * math.sqrt(6)) / 225],
        [(296 + 169 * math.sqrt(6)) / 1800, (88 + 7 * math.sqrt(6)) / 360, (-2 - 3 * math.sqrt(6)) / 225],
        [1 / 9, (16 + math.sqrt(6)) / 36, (16 - math.sqrt(6)) / 36],
    ], dtype=torch.float64),
    b=torch.tensor([1 / 9, (16 + math.sqrt(6)) / 36, (16 - math.sqrt(6)) / 36], dtype=torch.float64),
    c=torch.tensor([(4 - math.sqrt(6)) / 10, 1 / 2, (4 + math.sqrt(6)) / 10], dtype=torch.float64),
    order=5
)
