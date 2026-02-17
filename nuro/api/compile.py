"""Compiler entry point — compile() dispatches to target backends."""

from __future__ import annotations

from nuro.api.graph import Graph
from nuro.backends import get_backend
from nuro.backends.base import CompiledModel
from nuro.ir import IRGraph


def compile(
    graph: Graph,
    target: str = "auto",
    *,
    requires_grad: bool = False,
    surrogate: str = "atan",
) -> CompiledModel:
    """Compile a Graph to a runnable model on the specified backend.

    Parameters
    ----------
    graph : Graph
        The computation graph to compile.
    target : str
        Backend target. "auto" defaults to "gpu".
    requires_grad : bool
        When ``True``, the compiled model supports backpropagation through
        time (BPTT) using surrogate gradients.  Neurons become differentiable,
        the ``torch.no_grad()`` wrapper is removed from ``run()``, and
        ``run()`` returns an output spikes dict for loss computation.
    surrogate : str
        Surrogate gradient function name (``"atan"``, ``"sigmoid"``,
        ``"triangular"``).  Only used when ``requires_grad=True``.

    Returns
    -------
    CompiledModel
        A compiled model ready to run.
    """
    if target == "auto":
        target = "gpu"

    ir_graph = IRGraph.from_api_graph(graph)
    backend = get_backend(target)
    return backend.compile(
        ir_graph, requires_grad=requires_grad, surrogate=surrogate
    )
