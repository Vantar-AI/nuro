"""Surrogate gradient functions for differentiable spiking networks.

Spiking neurons use a Heaviside step function for spike generation, which has
zero gradient almost everywhere.  Surrogate gradients replace the backward
pass with a smooth approximation, enabling backpropagation-through-time (BPTT).

Usage::

    from nuro.backends.gpu.surrogates import SurrogateSpike, get_surrogate

    # In a neuron forward pass:
    spikes = SurrogateSpike.apply(v - v_thresh, surrogate_fn)

Available surrogate functions: ``"atan"``, ``"sigmoid"``, ``"triangular"``.
"""

from __future__ import annotations

import math

import torch


class SurrogateSpike(torch.autograd.Function):
    """Heaviside step forward, surrogate gradient backward.

    Parameters passed via ``apply(x, surrogate_fn)``:

    x : Tensor
        Membrane potential minus threshold (v - v_thresh).
    surrogate_fn : callable
        A function ``f(x) -> Tensor`` that returns the surrogate gradient
        evaluated at *x*.  See :func:`get_surrogate`.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, surrogate_fn) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.surrogate_fn = surrogate_fn
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_surrogate = ctx.surrogate_fn(x)
        return grad_output * grad_surrogate, None


# ---------------------------------------------------------------------------
# Built-in surrogate functions
# ---------------------------------------------------------------------------

def _atan_surrogate(alpha: float = 2.0):
    """ATan surrogate: ``1 / (1 + (pi * alpha * x)^2) * alpha``."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        return alpha / (1.0 + (math.pi * alpha * x) ** 2)
    return fn


def _sigmoid_surrogate(alpha: float = 4.0):
    """Sigmoid surrogate: ``alpha * sig(alpha * x) * (1 - sig(alpha * x))``."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(alpha * x)
        return alpha * s * (1.0 - s)
    return fn


def _triangular_surrogate(alpha: float = 1.0):
    """Triangular surrogate: ``max(0, 1/alpha - |x| / alpha^2)``."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(1.0 / alpha - x.abs() / (alpha ** 2), min=0.0)
    return fn


_SURROGATES = {
    "atan": _atan_surrogate,
    "sigmoid": _sigmoid_surrogate,
    "triangular": _triangular_surrogate,
}


def get_surrogate(name: str, **kwargs) -> callable:
    """Get a surrogate gradient function by name.

    Parameters
    ----------
    name : str
        One of ``"atan"``, ``"sigmoid"``, ``"triangular"``.
    **kwargs
        Passed to the surrogate constructor (e.g. ``alpha=2.0``).

    Returns
    -------
    callable
        A function ``f(x) -> Tensor`` suitable for :class:`SurrogateSpike`.
    """
    if name not in _SURROGATES:
        raise ValueError(
            f"Unknown surrogate '{name}'. Available: {sorted(_SURROGATES)}"
        )
    return _SURROGATES[name](**kwargs)
