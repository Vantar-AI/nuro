"""Tests for logging and callbacks."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import nuro
from nuro.callbacks import Callback, PrintCallback


class SpyCallback(Callback):
    """Test callback that records all invocations."""

    def __init__(self):
        self.events = []

    def on_run_start(self, config):
        self.events.append(("run_start", config))

    def on_step(self, step, spikes, metrics):
        self.events.append(("step", step))

    def on_run_end(self, metrics):
        self.events.append(("run_end", metrics))

    def close(self):
        self.events.append(("close",))


class TestCallbacks:
    """Test callback integration with GPU backend."""

    def test_callback_invoked(self):
        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2)
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")

        spy = SpyCallback()
        model.run(duration=0.005, dt=1e-3, callbacks=[spy])

        event_types = [e[0] for e in spy.events]
        assert "run_start" in event_types
        assert "run_end" in event_types
        assert event_types.count("step") == 5  # 5ms / 1ms

    def test_multiple_callbacks(self):
        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2)
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")

        spy1 = SpyCallback()
        spy2 = SpyCallback()
        model.run(duration=0.003, dt=1e-3, callbacks=[spy1, spy2])

        assert len(spy1.events) == len(spy2.events)

    def test_print_callback(self, capsys):
        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2)
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")

        cb = PrintCallback(log_interval=1)
        model.run(duration=0.003, dt=1e-3, callbacks=[cb])

        captured = capsys.readouterr()
        assert "[nuro]" in captured.out

    def test_no_callbacks_default(self):
        """Run without callbacks should work fine."""
        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2)
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")
        model.run(duration=0.003, dt=1e-3)
        assert model.metrics["num_steps"] == 3


class TestSynapticDelays:
    """Test synaptic delay functionality."""

    def test_zero_delay(self):
        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2, delay=0.0)
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")
        model.run(duration=0.01, dt=1e-3)
        assert model.metrics["num_steps"] == 10

    def test_nonzero_delay(self):
        pop1 = nuro.Population(size=4, dynamics="lif")
        pop2 = nuro.Population(size=3, dynamics="lif")
        conn = nuro.Connection(source=pop1, target=pop2, delay=3e-3)
        graph = nuro.Graph([pop1, pop2], [conn])
        model = nuro.compile(graph, target="gpu")
        model.run(duration=0.01, dt=1e-3)
        assert model.metrics["num_steps"] == 10

    def test_delay_suppresses_early_spikes(self):
        """With delay, output should spike later than without."""
        import torch

        pop1 = nuro.Population(size=4, dynamics="if")
        pop2 = nuro.Population(size=4, dynamics="if")

        # Strong static input that fires every step
        data = torch.ones(20, 4)
        inp = nuro.Input(population=pop1, data=data)

        # No delay
        conn_no_delay = nuro.Connection(source=pop1, target=pop2, delay=0.0)
        graph1 = nuro.Graph([pop1, pop2], [conn_no_delay], inputs=[inp])
        m1 = nuro.compile(graph1, target="gpu")
        m1.run(duration=0.02, dt=1e-3)
        spikes_no_delay = m1.metrics["total_spikes"]

        # With delay — first few steps should have fewer output spikes
        conn_delay = nuro.Connection(source=pop1, target=pop2, delay=5e-3)
        graph2 = nuro.Graph([pop1, pop2], [conn_delay], inputs=[inp])
        m2 = nuro.compile(graph2, target="gpu")
        m2.run(duration=0.02, dt=1e-3)
        spikes_delay = m2.metrics["total_spikes"]

        # Delayed model should have fewer or equal spikes
        assert spikes_delay <= spikes_no_delay
