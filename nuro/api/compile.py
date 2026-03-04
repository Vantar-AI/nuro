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
    quantize: bool | None = None,
    num_bits: int = 8,
    scale_factor: float = 1.0,
    quantize_aware: bool = False,
    online_learning: bool = False,
    # Cloud backend kwargs
    hardware: str = "loihi",
    api_key: str | None = None,
    endpoint: str | None = None,
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
    quantize : bool or None
        When ``True``, convert weights to fixed-point integers before
        writing to hardware.  When ``None`` (default), auto-quantization
        is enabled for hardware targets (loihi, spinnaker2) when
        ``weights_from`` is set.  Set to ``False`` to explicitly disable.
    num_bits : int
        Fixed-point precision for weight quantization.  Default 8
        (Loihi 2 native).  Only used when quantization is active.
    scale_factor : float
        Manual weight scale override.  When ``quantize=False``, weights
        are multiplied by this value.  Default 1.0.
    quantize_aware : bool
        When ``True`` and ``target="gpu"``, enable quantization-aware
        training with fake quantization during forward passes.
    online_learning : bool
        When ``True``, enable on-chip learning rules for hardware backends.
        On Loihi, uses ``LearningDense`` for STDP.  On SpiNNaker 2,
        enables ARM core learning rule execution.  Default ``False``.
    hardware : str
        Target chip for the cloud backend (``"loihi"`` or ``"spinnaker2"``).
        Only used when ``target="cloud"``.  Default ``"loihi"``.
    api_key : str, optional
        Vantar Cloud API key.  Falls back to ``VANTAR_API_KEY`` environment
        variable.  Only used when ``target="cloud"``.
    endpoint : str, optional
        Vantar Cloud API base URL override.  Defaults to
        ``https://api.vantar.xyz``.  Only used when ``target="cloud"``.

    Returns
    -------
    CompiledModel
        A compiled model ready to run.
    """
    if target == "auto":
        target = "gpu"

    # Auto-quantization: enable for hardware targets when weights are provided
    if quantize is None:
        if target in ("loihi", "spinnaker2") and weights_from is not None:
            quantize = True
        else:
            quantize = False

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
        quantize_aware=quantize_aware,
        online_learning=online_learning,
        hardware=hardware,
        api_key=api_key,
        endpoint=endpoint,
    )
