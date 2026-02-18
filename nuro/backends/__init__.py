"""Nuro backends — compilation targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nuro.backends.base import Backend

_REGISTRY: dict[str, str] = {
    "gpu": "nuro.backends.gpu.backend:GPUBackend",
    "loihi": "nuro.backends.loihi.backend:LoihiBackend",
    "spinnaker2": "nuro.backends.spinnaker2.backend:SpiNNaker2Backend",
    "cloud": "nuro.backends.cloud.backend:CloudBackend",
}


def get_backend(target: str) -> Backend:
    """Get a backend instance by name, with lazy imports."""
    if target not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{target}'. Available: {sorted(_REGISTRY)}"
        )
    module_path, class_name = _REGISTRY[target].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()
