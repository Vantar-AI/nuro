"""GPU → Akida weight transfer and quantization."""

from __future__ import annotations

import numpy as np


def load_gpu_weights(weights_from: str) -> dict[str, np.ndarray]:
    """Load synapse weights from a GPU checkpoint file.

    Parameters
    ----------
    weights_from : str
        Path to a ``.pt`` checkpoint.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of synapse key to weight matrix.
    """
    import torch

    checkpoint = torch.load(weights_from, weights_only=False, map_location="cpu")
    state_dict = checkpoint["weights"]

    weights = {}
    for key, tensor in state_dict.items():
        if key.startswith("synapses.") and key.endswith(".weight"):
            synapse_key = key[len("synapses."):-len(".weight")]
            weights[synapse_key] = tensor.numpy()

    return weights


def quantize_weights_akida(
    weights: np.ndarray,
    num_bits: int = 4,
) -> tuple[np.ndarray, float]:
    """Quantize weights for Akida hardware.

    Akida supports 1, 2, 4, or 8-bit weight quantization.
    4-bit is the default for best efficiency.

    Parameters
    ----------
    weights : np.ndarray
        Float weight matrix.
    num_bits : int
        Bit precision. Default 4.

    Returns
    -------
    quantized : np.ndarray
        Integer weight matrix.
    scale : float
        Scale factor applied.
    """
    max_int = 2 ** (num_bits - 1) - 1
    min_int = -(2 ** (num_bits - 1))

    max_abs = np.max(np.abs(weights))
    if max_abs == 0.0:
        return np.zeros_like(weights, dtype=np.int8), 1.0

    scale = max_int / max_abs
    scaled = weights * scale
    quantized = np.round(scaled).astype(np.int8)
    quantized = np.clip(quantized, min_int, max_int)

    return quantized, scale
