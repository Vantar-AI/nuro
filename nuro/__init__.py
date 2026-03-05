"""Nuro — The universal programming language for neuromorphic, thermodynamic, and biological computing."""

__version__ = "0.7.0"

from nuro.api.compile import compile
from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.input import Input
from nuro.api.objective import Objective
from nuro.api.population import Population
from nuro.backends.cloud.client import set_api_key
from nuro.calibration import CalibrationProfile
from nuro.experiment import Experiment
from nuro.recording import Recording
from nuro.sweep import ParameterSweep
from nuro import adapters, copilot, plot


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


def from_nir(nir_graph):
    """Import a NIR graph into Nuro.

    Parameters
    ----------
    nir_graph : nir.NIRGraph
        A NIR graph from any compatible framework.

    Returns
    -------
    IRGraph
        Nuro IR ready for ``compile()``.
    """
    from nuro.ir.nir_compat import from_nir as _from_nir

    return _from_nir(nir_graph)


def to_nir(graph):
    """Export a Nuro graph to NIR format.

    Parameters
    ----------
    graph : Graph or IRGraph
        A Nuro API Graph or IRGraph.

    Returns
    -------
    nir.NIRGraph
        NIR graph compatible with any NIR-supporting framework.
    """
    from nuro.ir import IRGraph
    from nuro.ir.nir_compat import to_nir as _to_nir

    if isinstance(graph, IRGraph):
        return _to_nir(graph)
    # API Graph → IR → NIR
    ir = IRGraph.from_api_graph(graph)
    return _to_nir(ir)


def convert_ann(model, input_shape, num_steps=100):
    """Convert a PyTorch ANN (MLP/CNN) to a Nuro SNN Graph.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model (Sequential or custom).
    input_shape : tuple
        Input tensor shape (excluding batch dim).
    num_steps : int
        Number of timesteps for rate coding.

    Returns
    -------
    Graph
        Nuro Graph ready for ``compile()``.
    """
    from nuro.conversion.ann2snn import convert_ann as _convert

    return _convert(model, input_shape, num_steps)


def experiment(name: str, **kwargs) -> Experiment:
    """Create a new Experiment (convenience factory)."""
    return Experiment(name, **kwargs)


__all__ = [
    "CalibrationProfile",
    "Connection",
    "Experiment",
    "Graph",
    "Input",
    "Objective",
    "ParameterSweep",
    "Population",
    "Recording",
    "adapters",
    "compile",
    "convert_ann",
    "copilot",
    "experiment",
    "from_nir",
    "load",
    "plot",
    "set_api_key",
    "to_nir",
]
