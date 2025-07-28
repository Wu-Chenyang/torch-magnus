"""
Torch Magnus Solvers

A PyTorch-based library for solving linear matrix ordinary differential equations
(ODEs) using Magnus-type integrators. This package provides efficient and
differentiable solvers suitable for problems in quantum mechanics, control theory,
and other areas of physics and engineering.

Available Functions:
- odeint: Solves the ODE at specified time points.
- odeint_adjoint: Solves the ODE with a memory-efficient adjoint method for backpropagation.
- adaptive_ode_solve: A lower-level solver for adaptive-step integration over a single interval.
"""

__version__ = "0.1.1"

from .solvers import (
    odeint,
    odeint_adjoint,
    adaptive_ode_solve,
    Magnus2nd,
    Magnus4th,
    Magnus6th,
    DenseOutputNaive,
    CollocationDenseOutput,
)

__all__ = [
    "odeint",
    "odeint_adjoint",
    "adaptive_ode_solve",
    "Magnus2nd",
    "Magnus4th",
    "Magnus6th",
    "DenseOutputNaive",
    "CollocationDenseOutput",
]
