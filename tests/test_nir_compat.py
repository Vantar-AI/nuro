"""Tests for NIR ↔ Nuro IR compatibility layer."""

from __future__ import annotations

import pytest
import numpy as np

nir = pytest.importorskip("nir")

from nuro.ir.nir_compat import from_nir, to_nir
from nuro.ir import IRGraph
from nuro.ir.nodes import DynamicsNode
from nuro.ir.edges import SynapticEdge


class TestFromNIR:
    """Test NIR → Nuro IR conversion."""

    def test_lif_network(self):
        """Import a simple LIF network from NIR."""
        nodes = {
            "input": nir.Input(input_type={"input": np.array([10])}),
            "lif1": nir.LIF(
                tau=np.full(10, 20e-3, dtype=np.float32),
                r=np.ones(10, dtype=np.float32),
                v_leak=np.zeros(10, dtype=np.float32),
                v_threshold=np.ones(10, dtype=np.float32),
            ),
            "linear": nir.Affine(
                weight=np.random.randn(5, 10).astype(np.float32),
                bias=np.zeros(5, dtype=np.float32),
            ),
            "lif2": nir.LIF(
                tau=np.full(5, 10e-3, dtype=np.float32),
                r=np.ones(5, dtype=np.float32),
                v_leak=np.zeros(5, dtype=np.float32),
                v_threshold=np.ones(5, dtype=np.float32),
            ),
            "output": nir.Output(output_type={"output": np.array([5])}),
        }
        edges = [
            ("input", "lif1"),
            ("lif1", "linear"),
            ("linear", "lif2"),
            ("lif2", "output"),
        ]
        nir_graph = nir.NIRGraph(nodes=nodes, edges=edges)

        ir = from_nir(nir_graph)

        assert isinstance(ir, IRGraph)
        # input, lif1, lif2, output (linear is absorbed as edge)
        assert "lif1" in ir.nodes
        assert "lif2" in ir.nodes
        assert ir.nodes["lif1"].dynamics == "lif"
        assert ir.nodes["lif2"].dynamics == "lif"
        assert ir.nodes["lif1"].size == 10
        assert ir.nodes["lif2"].size == 5

    def test_if_network(self):
        """Import an IF network from NIR."""
        nodes = {
            "n1": nir.IF(
                r=np.ones(8, dtype=np.float32),
                v_threshold=np.full(8, 1.0, dtype=np.float32),
            ),
            "n2": nir.IF(
                r=np.ones(4, dtype=np.float32),
                v_threshold=np.full(4, 1.0, dtype=np.float32),
            ),
        }
        edges = [("n1", "n2")]
        nir_graph = nir.NIRGraph(nodes=nodes, edges=edges)

        ir = from_nir(nir_graph)

        assert ir.nodes["n1"].dynamics == "if"
        assert ir.nodes["n2"].dynamics == "if"
        assert ir.nodes["n1"].size == 8
        assert len(ir.edges) >= 1

    def test_cubalif(self):
        """Import a CubaLIF node."""
        nodes = {
            "cuba": nir.CubaLIF(
                tau_mem=np.full(6, 15e-3, dtype=np.float32),
                tau_syn=np.full(6, 5e-3, dtype=np.float32),
                r=np.ones(6, dtype=np.float32),
                v_leak=np.zeros(6, dtype=np.float32),
                v_threshold=np.ones(6, dtype=np.float32),
                w_in=np.ones(6, dtype=np.float32),
            ),
        }
        nir_graph = nir.NIRGraph(nodes=nodes, edges=[])

        ir = from_nir(nir_graph)

        assert ir.nodes["cuba"].dynamics == "lif"
        assert abs(ir.nodes["cuba"].params["tau"] - 15e-3) < 1e-6

    def test_unsupported_node_raises(self):
        """Unsupported NIR node types raise ValueError."""
        nodes = {
            "delay": nir.Delay(delay=np.array([1.0])),
        }
        nir_graph = nir.NIRGraph(nodes=nodes, edges=[])

        with pytest.raises(ValueError, match="Unsupported NIR node type"):
            from_nir(nir_graph)


class TestToNIR:
    """Test Nuro IR → NIR conversion."""

    def test_lif_export(self):
        """Export a LIF IRGraph to NIR."""
        ir = IRGraph()
        n1 = DynamicsNode(id="pop1", size=10, dynamics="lif", params={"tau": 20e-3})
        n2 = DynamicsNode(id="pop2", size=5, dynamics="lif", params={"tau": 10e-3})
        ir.nodes["pop1"] = n1
        ir.nodes["pop2"] = n2
        ir._digraph.add_node("pop1", ir_node=n1)
        ir._digraph.add_node("pop2", ir_node=n2)
        edge = SynapticEdge(
            id="e1", source_id="pop1", target_id="pop2",
            pattern="dense", params={},
        )
        ir.edges.append(edge)
        ir._digraph.add_edge("pop1", "pop2", ir_edge=edge)

        nir_graph = to_nir(ir)

        assert "pop1" in nir_graph.nodes
        assert "pop2" in nir_graph.nodes
        assert isinstance(nir_graph.nodes["pop1"], nir.LIF)
        assert len(nir_graph.edges) >= 2  # src → affine → tgt

    def test_if_export(self):
        """Export an IF IRGraph to NIR."""
        ir = IRGraph()
        n1 = DynamicsNode(id="a", size=4, dynamics="if", params={})
        ir.nodes["a"] = n1
        ir._digraph.add_node("a", ir_node=n1)

        nir_graph = to_nir(ir)

        assert isinstance(nir_graph.nodes["a"], nir.IF)

    def test_unsupported_dynamics_raises(self):
        """Exporting unsupported dynamics raises ValueError."""
        ir = IRGraph()
        n = DynamicsNode(id="x", size=4, dynamics="izhikevich", params={})
        ir.nodes["x"] = n
        ir._digraph.add_node("x", ir_node=n)

        with pytest.raises(ValueError, match="Cannot export dynamics"):
            to_nir(ir)


class TestRoundTrip:
    """Test round-trip: Nuro → NIR → Nuro."""

    def test_lif_round_trip(self):
        """LIF network survives round-trip conversion."""
        ir = IRGraph()
        n1 = DynamicsNode(id="a", size=8, dynamics="lif", params={"tau": 20e-3, "v_thresh": 1.0})
        n2 = DynamicsNode(id="b", size=4, dynamics="lif", params={"tau": 10e-3, "v_thresh": 1.0})
        ir.nodes["a"] = n1
        ir.nodes["b"] = n2
        ir._digraph.add_node("a", ir_node=n1)
        ir._digraph.add_node("b", ir_node=n2)

        weights = np.random.randn(4, 8).astype(np.float32)
        edge = SynapticEdge(
            id="e1", source_id="a", target_id="b",
            pattern="dense", params={"weights": weights},
        )
        ir.edges.append(edge)
        ir._digraph.add_edge("a", "b", ir_edge=edge)

        # Nuro → NIR → Nuro
        nir_graph = to_nir(ir)
        ir2 = from_nir(nir_graph)

        assert "a" in ir2.nodes
        assert "b" in ir2.nodes
        assert ir2.nodes["a"].dynamics == "lif"
        assert ir2.nodes["b"].dynamics == "lif"
        assert ir2.nodes["a"].size == 8
        assert ir2.nodes["b"].size == 4

    def test_if_round_trip(self):
        """IF network survives round-trip."""
        ir = IRGraph()
        n = DynamicsNode(id="p", size=16, dynamics="if", params={"v_thresh": 1.0})
        ir.nodes["p"] = n
        ir._digraph.add_node("p", ir_node=n)

        nir_graph = to_nir(ir)
        ir2 = from_nir(nir_graph)

        assert ir2.nodes["p"].dynamics == "if"
        assert ir2.nodes["p"].size == 16
