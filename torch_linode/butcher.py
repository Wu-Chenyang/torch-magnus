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

    def clone(self, *args, **kwargs):
        """
        Creates a deep copy of the ButcherTableau instance.
        This method is consistent with the `torch.Tensor.clone()` API.

        All tensor attributes (c, b, a, b_error) are cloned, meaning new
        tensors are created with the same values. This operation is differentiable.

        Args:
            *args: Positional arguments passed to every tensor's `.clone()` method.
            **kwargs: Keyword arguments passed to every tensor's `.clone()` method.
                     (e.g., memory_format=torch.preserve_format)

        Returns:
            ButcherTableau: A new ButcherTableau instance with cloned tensors.
        """
        return ButcherTableau(
            c=self.c.clone(*args, **kwargs),
            b=self.b.clone(*args, **kwargs),
            a=self.a.clone(*args, **kwargs),
            order=self.order,
            b_error=self.b_error.clone(*args, **kwargs) if self.b_error is not None else None
        )

    def to(self, *args, **kwargs):
        """
        Performs ButcherTableau dtype and/or device conversion.
        This method is consistent with the `torch.Tensor.to()` API.

        Args:
            *args: Positional arguments passed to every tensor's `.to()` method.
            **kwargs: Keyword arguments passed to every tensor's `.to()` method.

        Returns:
            ButcherTableau: A new ButcherTableau instance with all tensors having the
                            specified dtype and/or device.
        """
        return ButcherTableau(
            c=self.c.to(*args, **kwargs),
            b=self.b.to(*args, **kwargs),
            a=self.a.to(*args, **kwargs),
            order=self.order,
            b_error=self.b_error.to(*args, **kwargs) if self.b_error is not None else None
        )

    def get_t_nodes(self, t0:torch.Tensor, h:torch.Tensor):
        self.c = self.c.to(device=t0.device, dtype=t0.dtype)
        if t0.ndim == 0 and h.ndim == 0:
            return t0 + self.c * h
        else:
            return (t0.unsqueeze(-1) + self.c * h.unsqueeze(-1)).reshape(-1)

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

# Implicit Runge-Kutta Methods - Gauss-Legendre
GL2 = ButcherTableau( # 1-stage, order 2 (Implicit Midpoint Rule)
    a=torch.tensor([[1/2]], dtype=torch.float64),
    b=torch.tensor([1], dtype=torch.float64),
    c=torch.tensor([1/2], dtype=torch.float64),
    order=2
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
