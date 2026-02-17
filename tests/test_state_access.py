"""Tests for state recording (voltages, spikes, weights)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.population import Population
from nuro.backends.gpu.backend import GPUBackend
from nuro.ir import IRGraph


def _make_model():
    p1 = Population(size=50, dynamics="lif", params={"tau": 20e-3})
    p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
    conn = Connection(source=p1, target=p2, pattern="dense", plasticity="stdp")
    graph = Graph([p1, p2], [conn])
    ir = IRGraph.from_api_graph(graph)
    model = GPUBackend().compile(ir)
    return model, p1, p2, conn


class TestSpikeRecording:
    def test_record_spikes(self):
        model, p1, p2, conn = _make_model()
        model.record("spikes", population=p2)
        model.run(duration=0.1, dt=1e-3)
        spikes = model.get_state("spikes", population=p2)
        assert spikes.shape == (100, 10)

    def test_record_source_spikes(self):
        model, p1, p2, conn = _make_model()
        model.record("spikes", population=p1)
        model.run(duration=0.05, dt=1e-3)
        spikes = model.get_state("spikes", population=p1)
        assert spikes.shape == (50, 50)


class TestVoltageRecording:
    def test_record_voltages(self):
        model, p1, p2, conn = _make_model()
        model.record("voltages", population=p2)
        model.run(duration=0.1, dt=1e-3)
        v = model.get_state("voltages", population=p2)
        # LIF neurons expose .v attribute
        assert v.shape[0] == 100
        assert v.shape[1] == 10


class TestWeightRecording:
    def test_record_weights_every_step(self):
        model, p1, p2, conn = _make_model()
        model.record("weights", connection=conn, interval=1)
        model.run(duration=0.01, dt=1e-3)
        w = model.get_state("weights", connection=conn)
        assert w.shape == (10, 10, 50)  # (steps, out, in)

    def test_record_weights_interval(self):
        model, p1, p2, conn = _make_model()
        model.record("weights", connection=conn, interval=5)
        model.run(duration=0.05, dt=1e-3)
        w = model.get_state("weights", connection=conn)
        assert w.shape[0] == 10  # 50 steps / 5 interval


class TestRecorderReset:
    def test_reset_clears_recordings(self):
        model, p1, p2, conn = _make_model()
        model.record("spikes", population=p2)
        model.run(duration=0.05, dt=1e-3)
        assert model.get_state("spikes", population=p2).shape[0] == 50

        model.reset()
        empty = model.get_state("spikes", population=p2)
        assert empty.numel() == 0

    def test_no_recording_by_default(self):
        """Without record() calls, zero overhead — no data stored."""
        model, p1, p2, conn = _make_model()
        model.run(duration=0.1, dt=1e-3)
        # get_state should return empty tensor for unregistered probes
        empty = model.get_state("spikes", population=p2)
        assert empty.numel() == 0
