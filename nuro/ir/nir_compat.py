"""NIR (Neuromorphic Intermediate Representation) compatibility layer.

Provides bidirectional conversion between Nuro IR and NIR format,
enabling interoperability with 9+ SNN frameworks and 5+ hardware platforms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nuro.ir import IRGraph
from nuro.ir.edges import SynapticEdge
from nuro.ir.nodes import DynamicsNode


def from_nir(nir_graph: Any) -> IRGraph:
    """Convert a NIR graph to a Nuro IRGraph.

    Parameters
    ----------
    nir_graph : nir.NIRGraph
        A NIR graph from any compatible framework (SpikingJelly, Norse,
        snnTorch, Sinabs, Lava, etc.).

    Returns
    -------
    IRGraph
        Nuro intermediate representation ready for ``compile()``.

    Raises
    ------
    ValueError
        If a NIR node type is not supported by Nuro.
    """
    import nir as nir_module

    ir = IRGraph()
    node_id_map: dict[str, str] = {}  # NIR node name → Nuro node id
    edge_counter = 0

    # First pass: convert NIR nodes to Nuro DynamicsNodes
    for name, node in nir_graph.nodes.items():
        if isinstance(node, nir_module.Input):
            # Input nodes become placeholder DynamicsNodes
            shape = node.input_type.get("input", [1])
            size = int(np.prod(shape)) if hasattr(shape, '__len__') else int(shape)
            dyn_node = DynamicsNode(
                id=name, size=max(size, 1), dynamics="if", params={}
            )
        elif isinstance(node, nir_module.Output):
            shape = node.output_type.get("output", [1])
            size = int(np.prod(shape)) if hasattr(shape, '__len__') else int(shape)
            dyn_node = DynamicsNode(
                id=name, size=max(size, 1), dynamics="if", params={}
            )
        elif isinstance(node, nir_module.LIF):
            size = int(np.prod(node.tau.shape)) if node.tau.shape else 1
            tau_val = float(np.mean(node.tau))
            v_thresh = float(np.mean(node.v_threshold))
            dyn_node = DynamicsNode(
                id=name,
                size=size,
                dynamics="lif",
                params={"tau": tau_val, "v_thresh": v_thresh},
            )
        elif isinstance(node, nir_module.IF):
            size = int(np.prod(node.r.shape)) if node.r.shape else 1
            v_thresh = float(np.mean(node.v_threshold))
            dyn_node = DynamicsNode(
                id=name,
                size=size,
                dynamics="if",
                params={"v_thresh": v_thresh},
            )
        elif isinstance(node, nir_module.CubaLIF):
            size = int(np.prod(node.tau_mem.shape)) if node.tau_mem.shape else 1
            tau_val = float(np.mean(node.tau_mem))
            v_thresh = float(np.mean(node.v_threshold))
            dyn_node = DynamicsNode(
                id=name,
                size=size,
                dynamics="lif",
                params={"tau": tau_val, "v_thresh": v_thresh},
            )
        elif isinstance(node, (nir_module.Affine, nir_module.Linear)):
            # Affine/Linear nodes become edges, but we need a placeholder node
            # They will be converted to SynapticEdges in the edge pass
            continue
        else:
            raise ValueError(
                f"Unsupported NIR node type '{type(node).__name__}' for node '{name}'. "
                f"Supported: Input, Output, LIF, IF, CubaLIF, Affine, Linear."
            )

        ir.nodes[name] = dyn_node
        ir._digraph.add_node(name, ir_node=dyn_node)
        node_id_map[name] = name

    # Second pass: convert NIR edges to Nuro SynapticEdges
    for src_name, tgt_name in nir_graph.edges:
        src_node_nir = nir_graph.nodes.get(src_name)
        tgt_node_nir = nir_graph.nodes.get(tgt_name)

        # Handle Affine/Linear nodes as synaptic connections
        if isinstance(tgt_node_nir, (nir_module.Affine, nir_module.Linear)):
            # This linear node sits between src and its downstream target
            # Find the downstream edges from this linear node
            downstream = [
                (s, t) for s, t in nir_graph.edges if s == tgt_name
            ]
            weights = tgt_node_nir.weight
            for _, final_tgt in downstream:
                if final_tgt in ir.nodes and src_name in ir.nodes:
                    edge_id = f"nir_edge_{edge_counter}"
                    edge_counter += 1
                    edge = SynapticEdge(
                        id=edge_id,
                        source_id=src_name,
                        target_id=final_tgt,
                        pattern="dense",
                        plasticity="none",
                        params={"weights": weights},
                    )
                    ir.edges.append(edge)
                    ir._digraph.add_edge(src_name, final_tgt, ir_edge=edge)
            continue

        if isinstance(src_node_nir, (nir_module.Affine, nir_module.Linear)):
            # Already handled above
            continue

        # Direct neuron-to-neuron edge (no weights)
        if src_name in ir.nodes and tgt_name in ir.nodes:
            edge_id = f"nir_edge_{edge_counter}"
            edge_counter += 1
            edge = SynapticEdge(
                id=edge_id,
                source_id=src_name,
                target_id=tgt_name,
                pattern="dense",
                plasticity="none",
                params={},
            )
            ir.edges.append(edge)
            ir._digraph.add_edge(src_name, tgt_name, ir_edge=edge)

    return ir


def to_nir(ir_graph: IRGraph) -> Any:
    """Convert a Nuro IRGraph to a NIR graph.

    Parameters
    ----------
    ir_graph : IRGraph
        Nuro intermediate representation.

    Returns
    -------
    nir.NIRGraph
        NIR graph compatible with any NIR-supporting framework.

    Raises
    ------
    ValueError
        If a Nuro dynamics type cannot be mapped to NIR.
    """
    import nir as nir_module

    nodes: dict[str, Any] = {}
    edges: list[tuple[str, str]] = []

    for nid, node in ir_graph.nodes.items():
        if node.dynamics == "lif":
            tau = node.params.get("tau", 20e-3)
            v_thresh = node.params.get("v_thresh", 1.0)
            nodes[nid] = nir_module.LIF(
                tau=np.full(node.size, tau, dtype=np.float32),
                r=np.ones(node.size, dtype=np.float32),
                v_leak=np.zeros(node.size, dtype=np.float32),
                v_threshold=np.full(node.size, v_thresh, dtype=np.float32),
            )
        elif node.dynamics == "if":
            v_thresh = node.params.get("v_thresh", 1.0)
            nodes[nid] = nir_module.IF(
                r=np.ones(node.size, dtype=np.float32),
                v_threshold=np.full(node.size, v_thresh, dtype=np.float32),
            )
        else:
            raise ValueError(
                f"Cannot export dynamics '{node.dynamics}' to NIR. "
                f"Supported: 'lif', 'if'."
            )

    for edge in ir_graph.edges:
        # Insert an Affine node for weighted connections
        weights = edge.params.get("weights")
        if weights is not None:
            affine_id = f"{edge.source_id}__affine__{edge.target_id}"
            src_size = ir_graph.nodes[edge.source_id].size
            tgt_size = ir_graph.nodes[edge.target_id].size
            w = np.array(weights, dtype=np.float32)
            if w.shape != (tgt_size, src_size):
                w = np.zeros((tgt_size, src_size), dtype=np.float32)
            nodes[affine_id] = nir_module.Affine(
                weight=w,
                bias=np.zeros(tgt_size, dtype=np.float32),
            )
            edges.append((edge.source_id, affine_id))
            edges.append((affine_id, edge.target_id))
        else:
            # No weights - insert identity affine
            src_size = ir_graph.nodes[edge.source_id].size
            tgt_size = ir_graph.nodes[edge.target_id].size
            affine_id = f"{edge.source_id}__affine__{edge.target_id}"
            nodes[affine_id] = nir_module.Affine(
                weight=np.eye(tgt_size, src_size, dtype=np.float32),
                bias=np.zeros(tgt_size, dtype=np.float32),
            )
            edges.append((edge.source_id, affine_id))
            edges.append((affine_id, edge.target_id))

    return nir_module.NIRGraph(nodes=nodes, edges=edges)
