"""GPU quantization utilities — QAT and post-training quantization.

Provides quantization-aware training (QAT) support for the GPU backend
and a unified interface for post-training quantization targeting different
hardware backends.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class FakeQuantize(torch.autograd.Function):
    """Straight-through estimator for fake quantization during training."""

    @staticmethod
    def forward(ctx, x, num_bits, min_val, max_val):
        scale = (max_val - min_val) / (2**num_bits - 1)
        if scale == 0:
            return x
        x_clamped = torch.clamp(x, min_val, max_val)
        x_quantized = torch.round((x_clamped - min_val) / scale) * scale + min_val
        return x_quantized

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through: pass gradients unchanged
        return grad_output, None, None, None


class QuantizedLinear(nn.Module):
    """Linear layer with fake quantization for quantization-aware training.

    Wraps ``nn.Linear`` and applies fake quantization to weights during
    forward passes, allowing the network to learn quantization-robust
    representations.

    Parameters
    ----------
    linear : nn.Linear
        The linear layer to wrap.
    num_bits : int
        Quantization bit-width. Default 8.
    target : str
        Target hardware for determining quantization range.
        ``"loihi"`` → [-256, 254], ``"spinnaker2"`` → [-15, 15].
    """

    def __init__(
        self,
        linear: nn.Linear,
        num_bits: int = 8,
        target: str = "loihi",
    ) -> None:
        super().__init__()
        self.linear = linear
        self.num_bits = num_bits
        self.target = target

        if target == "loihi":
            self.min_val = float(-(2**num_bits))
            self.max_val = float((2**num_bits) - 2)
        elif target == "spinnaker2":
            self.min_val = -15.0
            self.max_val = 15.0
        else:
            self.min_val = float(-(2 ** (num_bits - 1)))
            self.max_val = float(2 ** (num_bits - 1) - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fake-quantize weights
        w_scale = self.max_val / (torch.max(torch.abs(self.linear.weight)).item() + 1e-8)
        w_scaled = self.linear.weight * w_scale
        w_fake_q = FakeQuantize.apply(w_scaled, self.num_bits, self.min_val, self.max_val)
        w_dequant = w_fake_q / w_scale

        return nn.functional.linear(x, w_dequant, self.linear.bias)


def enable_qat(snn: nn.Module, num_bits: int = 8, target: str = "loihi") -> None:
    """Enable quantization-aware training on a NuroSNN module.

    Replaces all ``nn.Linear`` synapse layers with ``QuantizedLinear``
    wrappers that apply fake quantization during training.

    Parameters
    ----------
    snn : nn.Module
        The NuroSNN module (from ``GPUCompiledModel.snn``).
    num_bits : int
        Quantization bit-width.
    target : str
        Target hardware: ``"loihi"`` or ``"spinnaker2"``.
    """
    if hasattr(snn, "synapses"):
        for key in list(snn.synapses.keys()):
            layer = snn.synapses[key]
            if isinstance(layer, nn.Linear):
                snn.synapses[key] = QuantizedLinear(layer, num_bits, target)


def quantize_model(
    weights: dict[str, np.ndarray],
    target: str = "loihi",
    num_bits: int = 8,
) -> dict[str, np.ndarray]:
    """Post-training quantization for any hardware target.

    Unified interface that dispatches to the appropriate quantization
    scheme based on target hardware.

    Parameters
    ----------
    weights : dict[str, np.ndarray]
        Mapping of synapse key → float weight matrix.
    target : str
        ``"loihi"`` or ``"spinnaker2"``.
    num_bits : int
        Bit precision for quantization.

    Returns
    -------
    dict[str, np.ndarray]
        Quantized weight matrices.
    """
    result = {}
    for key, w in weights.items():
        if target == "loihi":
            from nuro.backends.loihi.transfer import quantize_weights
            q, _ = quantize_weights(w, num_bits=num_bits)
            result[key] = q
        elif target == "spinnaker2":
            from nuro.backends.spinnaker2.transfer import quantize_weights_s2
            q, _ = quantize_weights_s2(w)
            result[key] = q
        else:
            result[key] = w
    return result
