"""Nuro — The universal programming language for neuromorphic, thermodynamic, and biological computing."""

__version__ = "0.6.0"

from nuro.api.compile import compile
from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.input import Input
from nuro.api.objective import Objective
from nuro.api.population import Population
from nuro.backends.cloud.client import set_api_key
from nuro import copilot


def load(path: str, target: str = "gpu", dt: float = 1e-3):
    """Load a compiled model from a checkpoint file.

    Parameters
    ----------
    path : str
        Path to the checkpoint file.
    target : str
        Backend target (currently only ``"gpu"``).
    dt : float
        Simulation timestep.

    Returns
    -------
    CompiledModel
        A ready-to-run model with restored weights.
    """
    from nuro.backends.gpu.checkpoint import load_checkpoint

    return load_checkpoint(path, target=target, dt=dt)


__all__ = [
    "Connection",
    "Graph",
    "Input",
    "Objective",
    "Population",
    "compile",
    "copilot",
    "load",
    "set_api_key",
]
