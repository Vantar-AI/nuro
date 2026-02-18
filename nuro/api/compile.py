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
    weights_from: str | None = None,
    quantize: bool = False,
    num_bits: int = 8,
    scale_factor: float = 1.0,
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
    weights_from : str, optional
        Path to a GPU checkpoint file (``.pt``).  When provided, trained
        weights are loaded and transferred to the target backend.  This
        enables the train-on-GPU → deploy-to-hardware workflow.
    quantize : bool
        When ``True``, convert weights to fixed-point integers before
        writing to the Loihi backend.  Required for real Loihi 2 hardware;
        not needed for simulation.  Default ``False``.
    num_bits : int
        Fixed-point precision for weight quantization.  Default 8
        (Loihi 2 native).  Only used when ``quantize=True``.
    scale_factor : float
        Manual weight scale override.  When ``quantize=False``, weights
        are multiplied by this value.  Default 1.0.

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
        ir_graph,
        requires_grad=requires_grad,
        surrogate=surrogate,
        weights_from=weights_from,
        quantize=quantize,
        num_bits=num_bits,
        scale_factor=scale_factor,
    )
