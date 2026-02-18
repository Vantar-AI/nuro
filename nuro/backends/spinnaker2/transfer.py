"""GPU → SpiNNaker 2 weight transfer and quantization."""

from __future__ import annotations

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
        if key.startswith("synapses.") and key.endswith(".weight"):
            synapse_key = key[len("synapses."):-len(".weight")]
            weights[synapse_key] = tensor.numpy()

    return weights


def quantize_weights_s2(
    weights: np.ndarray,
    scale_factor: float | None = None,
) -> tuple[np.ndarray, float]:
    """Quantize float weights to SpiNNaker 2 integer range [-15, 15].

    SpiNNaker 2 synaptic weights are 8-bit signed integers with an
    effective usable range of [-15, 15] (hardware constraint).

    Parameters
    ----------
    weights : np.ndarray
        Floating-point weight matrix.
    scale_factor : float, optional
        Override the auto-computed scale. When ``None``, the scale is set
        so ``max(|weights|)`` maps to 15.

    Returns
    -------
    quantized : np.ndarray
        Integer weight matrix (dtype int32), values in [-15, 15].
    scale : float
        The scale factor applied (``quantized ≈ weights * scale``).
    """
    max_int = 15
    min_int = -15

    if scale_factor is None:
        max_abs = np.max(np.abs(weights))
        if max_abs == 0.0:
            return np.zeros_like(weights, dtype=np.int32), 1.0
        scale_factor = max_int / max_abs

    scaled = weights * scale_factor
    quantized = np.round(scaled).astype(np.int32)
    quantized = np.clip(quantized, min_int, max_int)

    return quantized, scale_factor


def weights_to_connection_list(
    weight_matrix: np.ndarray,
    delay: int = 0,
    threshold: float = 0.0,
) -> list[list[int | float]]:
    """Convert a weight matrix to SpiNNaker 2 connection list format.

    py-spinnaker2 ``Projection`` takes a list of
    ``[pre_idx, post_idx, weight, delay]`` tuples.

    Parameters
    ----------
    weight_matrix : np.ndarray
        Shape ``(post, pre)`` — matches GPU (out_features, in_features).
    delay : int
        Synaptic delay in timesteps. Hardware range: [0, 7]. Default 0.
    threshold : float
        Weights with absolute value below this threshold are pruned.
        Default 0.0 (keep all).

    Returns
    -------
    list of [pre_idx, post_idx, weight, delay]
    """
    delay = max(0, min(7, int(delay)))  # clamp to hardware range
    n_post, n_pre = weight_matrix.shape
    connections = []
    for post_idx in range(n_post):
        for pre_idx in range(n_pre):
            w = weight_matrix[post_idx, pre_idx]
            if abs(w) > threshold:
                connections.append([pre_idx, post_idx, float(w), delay])
    return connections
