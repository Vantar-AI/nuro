"""IRGraph → JSON serialization for Vantar Cloud API."""

from __future__ import annotations

from typing import Any

from nuro.ir import IRGraph


def serialize_ir_graph(ir_graph: IRGraph) -> dict[str, Any]:
    """Serialize an IRGraph to a JSON-compatible dict.

    The wire format is deliberately simple:
    - nodes: list of DynamicsNode dicts (id, size, dynamics, params)
    - edges: list of SynapticEdge dicts (id, source_id, target_id, pattern, plasticity)
    - inputs: list of input descriptor dicts (population_id, mode, rate, shape)

    Args:
        ir_graph: The IRGraph to serialize.

    Returns:
        JSON-compatible dict suitable for POST /v1/compile.
    """
    nodes = []
    for node in ir_graph.nodes:
        nodes.append({
            "id": node.id,
            "size": node.size,
            "dynamics": node.dynamics,
            "params": node.params or {},
        })

    edges = []
    for edge in ir_graph.edges:
        edges.append({
            "id": edge.id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "pattern": edge.pattern,
            "plasticity": edge.plasticity,
            "weight_scale": getattr(edge, "weight_scale", 1.0),
            "sparsity": getattr(edge, "sparsity", 0.1),
        })

    inputs = []
    for inp in getattr(ir_graph, "inputs", []):
        entry: dict[str, Any] = {"population_id": inp.population_id}
        if hasattr(inp, "mode") and inp.mode:
            entry["mode"] = inp.mode
            entry["rate"] = getattr(inp, "rate", 100.0)
        elif hasattr(inp, "shape"):
            entry["mode"] = "static"
            entry["shape"] = list(inp.shape)
        else:
            entry["mode"] = "poisson"
            entry["rate"] = 100.0
        inputs.append(entry)

    return {
        "version": "1",
        "nodes": nodes,
        "edges": edges,
        "inputs": inputs,
    }
