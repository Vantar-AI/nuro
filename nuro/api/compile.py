"""Compiler entry point — compile() dispatches to target backends."""

from __future__ import annotations

from nuro.api.graph import Graph
from nuro.backends import get_backend
from nuro.backends.base import CompiledModel
from nuro.ir import IRGraph


def compile(graph: Graph, target: str = "auto") -> CompiledModel:
    """Compile a Graph to a runnable model on the specified backend.

    Parameters
    ----------
    graph : Graph
        The computation graph to compile.
    target : str
        Backend target. "auto" defaults to "gpu".

    Returns
    -------
    CompiledModel
        A compiled model ready to run.
    """
    if target == "auto":
        target = "gpu"

    ir_graph = IRGraph.from_api_graph(graph)
    backend = get_backend(target)
    return backend.compile(ir_graph)
