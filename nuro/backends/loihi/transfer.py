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


def apply_weights_to_lava(
    weights: dict[str, np.ndarray],
    synapses: dict[str, Any],
    scale_factor: float = 1.0,
) -> None:
    """Apply loaded GPU weights to Lava Dense Processes.

    Parameters
    ----------
    weights : dict[str, np.ndarray]
        Mapping of synapse key → weight matrix from :func:`load_gpu_weights`.
    synapses : dict[str, Any]
        Mapping of synapse key → Lava Dense Process.
    scale_factor : float
        Multiplicative factor for weights (useful for fixed-point conversion
        on hardware). Default 1.0 (no scaling).
    """
    for key, weight_matrix in weights.items():
        if key in synapses:
            scaled = weight_matrix * scale_factor
            # Both GPU (nn.Linear) and Lava (Dense) use (out, in) format
            synapses[key].weights.init = scaled
