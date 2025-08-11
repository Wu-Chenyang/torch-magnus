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

GL8 = ButcherTableau(
    a=torch.tensor([
        [0.0869637112843634, -0.0266041800849988, 0.0126274626894047, -0.0035551496857957],
        [0.1881181174998673, 0.1630362887156379, -0.0278804286024718, 0.0067355005945384],
        [0.1671919219741852, 0.3539530060337503, 0.1630362887156333, -0.0141906949311409],
        [0.1774825722545171, 0.3134451147418754, 0.3526767575162733, 0.0869637112843601],
    ], dtype=torch.float64),
    b=torch.tensor([0.1739274225687268, 0.3260725774312732, 0.3260725774312732, 0.1739274225687268], dtype=torch.float64),
    c=torch.tensor([0.0694318442029737, 0.3300094782075719, 0.6699905217924281, 0.9305681557970262], dtype=torch.float64),
    order=8
)

GL10 = ButcherTableau(
    a=torch.tensor([
        [0.0592317212640472, -0.0195703643590760, 0.0112544008186429, -0.0055937936608122, 0.0015881129678660],
        [0.1281510056700447, 0.1196571676248428, -0.0245921146196431, 0.0103182806706838, -0.0027689943987697],
        [0.1137762880042217, 0.2600046516806465, 0.1422222222222197, -0.0206903164309578, 0.0046871545238698],
        [0.1212324369268592, 0.2289960545790058, 0.3090365590640876, 0.1196571676248400, -0.0096875631419511],
        [0.1168753295602220, 0.2449081289105025, 0.2731900436258014, 0.2588846996087633, 0.0592317212640426],
    ], dtype=torch.float64),
    b=torch.tensor([0.1184634425280945, 0.2393143352496832, 0.2844444444444444, 0.2393143352496832, 0.1184634425280945], dtype=torch.float64),
    c=torch.tensor([0.0469100770306680, 0.2307653449471584, 0.5000000000000000, 0.7692346550528415, 0.9530899229693320], dtype=torch.float64),
    order=10
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

# Gauss-Kronrod 15-point rule
_GK15_NODES_RAW = [-0.99145537112081263920685469752598, -0.94910791234275852452618968404809, -0.86486442335976907278971278864098, -0.7415311855993944398638647732811, -0.58608723546769113029414483825842, -0.40584515137739716690660641207707, -0.20778495500789846760068940377309, 0.0]
_GK15_WEIGHTS_K_RAW = [0.022935322010529224963732008059913, 0.063092092629978553290700663189093, 0.10479001032225018383987632254189, 0.14065325971552591874518959051021, 0.16900472663926790282658342659795, 0.19035057806478540991325640242055, 0.20443294007529889241416199923466, 0.20948214108472782801299917489173]
_GK15_WEIGHTS_G_RAW = [0.12948496616886969327061143267787, 0.2797053914892766679014677714229, 0.38183005050511894495036977548818, 0.41795918367346938775510204081658]

_nodes_neg = torch.tensor(_GK15_NODES_RAW, dtype=torch.float64)
_nodes = torch.cat([-torch.flip(_nodes_neg[0:-1], dims=[0]), _nodes_neg])
_weights_k_half = torch.tensor(_GK15_WEIGHTS_K_RAW, dtype=torch.float64)
_weights_k = torch.cat([torch.flip(_weights_k_half[0:-1], dims=[0]), _weights_k_half])
_weights_g_half = torch.tensor(_GK15_WEIGHTS_G_RAW, dtype=torch.float64)
_weights_g_embedded = torch.cat([torch.flip(_weights_g_half, dims=[0]), _weights_g_half[1:]])
_weights_g = torch.zeros_like(_weights_k)
_weights_g[1::2] = _weights_g_embedded

GK15 = ButcherTableau(
    c=(_nodes + 1) / 2,
    b=_weights_k / 2,
    a=torch.zeros((15, 15), dtype=torch.float64), # Not used for quadrature
    order=15, # Order of the Kronrod rule
    b_error=(_weights_k - _weights_g) / 2
)
