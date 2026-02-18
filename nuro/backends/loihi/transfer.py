"""GPU → Loihi weight transfer — load GPU checkpoint and apply to Lava Dense Processes."""

from __future__ import annotations

from typing import Any

import numpy as np


def load_gpu_weights(weights_from: str) -> dict[str, np.ndarray]:
    """Load synapse weights from a GPU checkpoint file.

    Parameters
    ----------
    weights_from : str
        Path to a ``.pt`` checkpoint saved by ``GPUCompiledModel.save()``.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of synapse key (``"src_id__tgt_id"``) to weight matrix
        as numpy array with shape ``(out_features, in_features)``.
    """
    import torch

    checkpoint = torch.load(weights_from, weights_only=False, map_location="cpu")
    state_dict = checkpoint["weights"]

    weights = {}
    for key, tensor in state_dict.items():
        # State dict keys are like "synapses.src__tgt.weight"
        if key.startswith("synapses.") and key.endswith(".weight"):
            synapse_key = key[len("synapses."):-len(".weight")]
            weights[synapse_key] = tensor.numpy()

    return weights


def quantize_weights(
    weights: np.ndarray,
    num_bits: int = 8,
    scale_factor: float | None = None,
) -> tuple[np.ndarray, float]:
    """Quantize floating-point weights to fixed-point integers for Loihi.

    Loihi 2 synaptic weights are stored as signed integers. The hardware
    uses ``num_bits`` precision (default 8-bit: range [-256, 254] step 2).
    A ``weight_exp`` exponent scales the integer back to float on-chip:
    ``effective_weight = int_weight * 2^weight_exp``.

    This function finds the optimal scale so that the largest weight maps
    to the maximum integer value, then clamps and rounds.

    Parameters
    ----------
    weights : np.ndarray
        Floating-point weight matrix (any shape).
    num_bits : int
        Bit precision for the weight integer. Default 8.
        Loihi uses signed integers so the usable range is
        ``[-(2^num_bits), 2^num_bits - 2]`` in steps of 2.
    scale_factor : float, optional
        Override the auto-computed scale. When ``None`` (default), the scale
        is set so ``max(|weights|)`` maps to the maximum integer value.

    Returns
    -------
    quantized : np.ndarray
        Integer weight matrix (same shape, dtype int32).
    scale : float
        The scale factor applied (``quantized ≈ weights * scale``).
        Store this to dequantize: ``weights_float ≈ quantized / scale``.
    """
    # Loihi signed weight range: -2^n to 2^n-2, step 2
    max_int = (2 ** num_bits) - 2          # e.g. 254 for 8-bit
    min_int = -(2 ** num_bits)             # e.g. -256 for 8-bit

    if scale_factor is None:
        max_abs = np.max(np.abs(weights))
        if max_abs == 0.0:
            return np.zeros_like(weights, dtype=np.int32), 1.0
        scale_factor = max_int / max_abs

    scaled = weights * scale_factor

    # Round to nearest even integer and clamp to valid range
    quantized = np.round(scaled).astype(np.int32)
    quantized = np.clip(quantized, min_int, max_int)

    # Loihi requires weights to be even integers (LSB is always 0)
    quantized = (quantized >> 1) << 1

    return quantized, scale_factor


def apply_weights_to_lava(
    weights: dict[str, np.ndarray],
    synapses: dict[str, Any],
    scale_factor: float = 1.0,
    quantize: bool = False,
    num_bits: int = 8,
) -> None:
    """Apply loaded GPU weights to Lava Dense Processes.

    Parameters
    ----------
    weights : dict[str, np.ndarray]
        Mapping of synapse key → weight matrix from :func:`load_gpu_weights`.
    synapses : dict[str, Any]
        Mapping of synapse key → Lava Dense Process.
    scale_factor : float
        Multiplicative scale applied before optional quantization.
        When ``quantize=False``, this is the only transformation.
        When ``quantize=True``, the auto-scale overrides this unless you
        set ``quantize=False`` and pass a manual scale.
    quantize : bool
        When ``True``, convert weights to fixed-point integers using
        :func:`quantize_weights` before writing to Lava Dense.
        Use this when targeting real Loihi 2 hardware (not simulation).
        Default ``False`` — keeps float weights for simulation.
    num_bits : int
        Fixed-point precision bits. Only used when ``quantize=True``.
        Default 8 (Loihi 2 native precision).
    """
    for key, weight_matrix in weights.items():
        if key not in synapses:
            continue

        scaled = weight_matrix * scale_factor

        if quantize:
            final_weights, _ = quantize_weights(scaled, num_bits=num_bits)
        else:
            final_weights = scaled

        # Both GPU (nn.Linear) and Lava (Dense) use (out, in) format
        synapses[key].weights.init = final_weights
