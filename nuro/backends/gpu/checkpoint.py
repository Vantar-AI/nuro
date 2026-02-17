"""GPU checkpointing — save and load compiled models."""

from __future__ import annotations

import warnings
from typing import Any

import torch

import nuro


def save_checkpoint(
    model: Any,
    path: str,
) -> None:
    """Save a GPUCompiledModel to disk.

    Stores the SNN weights, the IR graph, and SDK metadata.

    Parameters
    ----------
    model : GPUCompiledModel
        The compiled model to save.
    path : str
        File path for the checkpoint (e.g. ``"checkpoint.pt"``).
    """
    checkpoint = {
        "weights": model._snn.state_dict(),
        "ir": model._ir_graph,
        "version": nuro.__version__,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    target: str = "gpu",
    dt: float = 1e-3,
) -> Any:
    """Load a GPUCompiledModel from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    target : str
        Backend target (currently only ``"gpu"``).
    dt : float
        Timestep for the reconstructed model.

    Returns
    -------
    GPUCompiledModel
        A ready-to-run model with restored weights.
    """
    from nuro.backends.gpu.backend import GPUBackend, NuroSNN

    checkpoint = torch.load(path, weights_only=False)

    saved_version = checkpoint.get("version", "unknown")
    if saved_version != nuro.__version__:
        warnings.warn(
            f"Checkpoint was saved with nuro {saved_version}, "
            f"but current version is {nuro.__version__}. "
            "This may cause compatibility issues.",
            stacklevel=2,
        )

    ir_graph = checkpoint["ir"]
    snn = NuroSNN(ir_graph, dt)
    snn.load_state_dict(checkpoint["weights"])

    from nuro.backends.gpu.backend import GPUCompiledModel

    model = GPUCompiledModel(snn, ir_graph)
    return model
