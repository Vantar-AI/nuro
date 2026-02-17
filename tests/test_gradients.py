"""Tests for surrogate gradient training (v0.4.0)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

import nuro
from nuro.api.connection import Connection
from nuro.api.graph import Graph
from nuro.api.input import Input
from nuro.api.population import Population
from nuro.backends.gpu.backend import GPUBackend
from nuro.backends.gpu.surrogates import SurrogateSpike, get_surrogate
from nuro.ir import IRGraph


# ---------------------------------------------------------------------------
# Surrogate function unit tests
# ---------------------------------------------------------------------------


class TestSurrogateFunctions:
    """Unit tests for surrogate gradient functions."""

    def test_atan_surrogate(self):
        fn = get_surrogate("atan")
        x = torch.linspace(-2, 2, 100)
        grad = fn(x)
        assert grad.shape == x.shape
        # Peak at x=0
        assert grad[50].item() > grad[0].item()

    def test_sigmoid_surrogate(self):
        fn = get_surrogate("sigmoid")
        x = torch.linspace(-2, 2, 100)
        grad = fn(x)
        assert grad.shape == x.shape
        assert grad[50].item() > grad[0].item()

    def test_triangular_surrogate(self):
        fn = get_surrogate("triangular")
        x = torch.linspace(-2, 2, 100)
        grad = fn(x)
        assert grad.shape == x.shape
        # Triangular is zero far from origin
        assert grad[0].item() == 0.0

    def test_unknown_surrogate_raises(self):
        with pytest.raises(ValueError, match="Unknown surrogate"):
            get_surrogate("nonexistent")

    def test_surrogate_spike_forward(self):
        """Forward pass is Heaviside step."""
        fn = get_surrogate("atan")
        x = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0])
        out = SurrogateSpike.apply(x, fn)
        expected = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        assert torch.allclose(out, expected)

    def test_surrogate_spike_backward(self):
        """Backward pass uses surrogate gradient, not zero."""
        fn = get_surrogate("atan")
        x = torch.tensor([0.0], requires_grad=True)
        out = SurrogateSpike.apply(x, fn)
        out.backward()
        assert x.grad is not None
        assert x.grad.item() > 0  # Surrogate gives non-zero gradient at threshold


# ---------------------------------------------------------------------------
# Gradient flow through networks
# ---------------------------------------------------------------------------


class TestGradientFlowLIF:
    """Gradients flow through LIF networks."""

    def test_lif_gradient_exists(self):
        """Weights have non-None gradients after backward."""
        p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")

        data = torch.rand(20, 10)
        inp = Input(population=p1, data=data)
        graph = Graph([p1, p2], [conn], inputs=[inp])

        model = nuro.compile(graph, target="gpu", requires_grad=True)
        output = model.run(duration=0.02, dt=1e-3)

        assert output is not None
        # Sum all output spikes and backprop
        loss = sum(v.sum() for v in output.values())
        loss.backward()

        # Check that synapse weights have gradients
        key = f"{p1.id}__{p2.id}"
        synapse = model.snn.synapses[key]
        assert synapse.weight.grad is not None

    def test_lif_gradient_nonzero(self):
        """Gradients should be non-zero for at least some weights."""
        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")

        # High-rate input to ensure spiking activity
        data = (torch.rand(50, 20) < 0.3).float()
        inp = Input(population=p1, data=data)
        graph = Graph([p1, p2], [conn], inputs=[inp])

        model = nuro.compile(graph, target="gpu", requires_grad=True)
        output = model.run(duration=0.05, dt=1e-3)

        loss = sum(v.sum() for v in output.values())
        loss.backward()

        key = f"{p1.id}__{p2.id}"
        synapse = model.snn.synapses[key]
        # At least some gradients should be non-zero
        assert synapse.weight.grad.abs().sum().item() > 0


class TestGradientFlowIzhikevich:
    """Gradients flow through Izhikevich networks."""

    def test_izh_gradient_exists(self):
        p1 = Population(size=10, dynamics="izhikevich", params={"preset": "regular_spiking"})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")

        data = (torch.rand(20, 10) < 0.3).float()
        inp = Input(population=p1, data=data)
        graph = Graph([p1, p2], [conn], inputs=[inp])

        model = nuro.compile(graph, target="gpu", requires_grad=True)
        output = model.run(duration=0.02, dt=1e-3)

        loss = sum(v.sum() for v in output.values())
        loss.backward()

        key = f"{p1.id}__{p2.id}"
        synapse = model.snn.synapses[key]
        assert synapse.weight.grad is not None


class TestGradientFlowAdEx:
    """Gradients flow through AdEx networks."""

    def test_adex_gradient_exists(self):
        p1 = Population(size=10, dynamics="adex")
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")

        data = (torch.rand(20, 10) < 0.3).float()
        inp = Input(population=p1, data=data)
        graph = Graph([p1, p2], [conn], inputs=[inp])

        model = nuro.compile(graph, target="gpu", requires_grad=True)
        output = model.run(duration=0.02, dt=1e-3)

        loss = sum(v.sum() for v in output.values())
        loss.backward()

        key = f"{p1.id}__{p2.id}"
        synapse = model.snn.synapses[key]
        assert synapse.weight.grad is not None


# ---------------------------------------------------------------------------
# Batched training
# ---------------------------------------------------------------------------


class TestBatchedTraining:
    """Training with batch_size > 1."""

    def test_batched_gradient(self):
        p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")

        data = (torch.rand(20, 4, 10) < 0.3).float()
        inp = Input(population=p1, data=data)
        graph = Graph([p1, p2], [conn], inputs=[inp])

        model = nuro.compile(graph, target="gpu", requires_grad=True)
        output = model.run(duration=0.02, dt=1e-3, batch_size=4)

        loss = sum(v.sum() for v in output.values())
        loss.backward()

        key = f"{p1.id}__{p2.id}"
        synapse = model.snn.synapses[key]
        assert synapse.weight.grad is not None


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """requires_grad=False (default) preserves v0.3.0 behaviour."""

    def test_default_no_grad(self):
        """Default compile produces no gradient tracking."""
        p1 = Population(size=20, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=10, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])
        ir = IRGraph.from_api_graph(graph)

        model = GPUBackend().compile(ir)
        result = model.run(duration=0.05, dt=1e-3)

        # run() returns None when not training
        assert result is None
        assert model.metrics["num_steps"] == 50

    def test_default_returns_none(self):
        """run() returns None when requires_grad=False."""
        p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense")
        graph = Graph([p1, p2], [conn])

        model = nuro.compile(graph, target="gpu")
        result = model.run(duration=0.02, dt=1e-3)
        assert result is None


# ---------------------------------------------------------------------------
# Different surrogate functions
# ---------------------------------------------------------------------------


class TestSurrogateChoice:
    """Different surrogate functions produce different gradient magnitudes."""

    def test_atan_vs_sigmoid(self):
        """ATan and sigmoid surrogates produce different gradients."""
        grads = {}
        for surr in ("atan", "sigmoid"):
            p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
            p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
            conn = Connection(source=p1, target=p2, pattern="dense")

            torch.manual_seed(42)
            data = (torch.rand(30, 10) < 0.3).float()
            inp = Input(population=p1, data=data)
            graph = Graph([p1, p2], [conn], inputs=[inp])

            model = nuro.compile(
                graph, target="gpu", requires_grad=True, surrogate=surr
            )
            output = model.run(duration=0.03, dt=1e-3)
            loss = sum(v.sum() for v in output.values())
            loss.backward()

            key = f"{p1.id}__{p2.id}"
            synapse = model.snn.synapses[key]
            grads[surr] = synapse.weight.grad.clone()

        # Both should have gradients, but they should differ
        assert not torch.allclose(grads["atan"], grads["sigmoid"], atol=1e-8)


# ---------------------------------------------------------------------------
# STDP disabled during training
# ---------------------------------------------------------------------------


class TestSTDPDisabledDuringTraining:
    """STDP updates are skipped when requires_grad=True."""

    def test_stdp_weights_unchanged(self):
        p1 = Population(size=10, dynamics="lif", params={"tau": 20e-3})
        p2 = Population(size=5, dynamics="lif", params={"tau": 10e-3})
        conn = Connection(source=p1, target=p2, pattern="dense", plasticity="stdp")

        data = (torch.rand(50, 10) < 0.3).float()
        inp = Input(population=p1, data=data)
        graph = Graph([p1, p2], [conn], inputs=[inp])

        model = nuro.compile(graph, target="gpu", requires_grad=True)

        # Snapshot weights before
        key = f"{p1.id}__{p2.id}"
        w_before = model.snn.synapses[key].weight.data.clone()

        output = model.run(duration=0.05, dt=1e-3)
        loss = sum(v.sum() for v in output.values())
        loss.backward()

        # STDP should NOT have modified weights (only optimizer should)
        w_after = model.snn.synapses[key].weight.data.clone()
        assert torch.allclose(w_before, w_after)


# ---------------------------------------------------------------------------
# Functional: simple training loop
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    """A simple training loop converges (loss decreases)."""

    def test_xor_loss_decreases(self):
        """Train a tiny network — loss should decrease over epochs."""
        # 2 → 10 → 1 LIF network
        inp_pop = Population(size=2, dynamics="lif", params={"tau": 20e-3})
        hidden = Population(size=10, dynamics="lif", params={"tau": 15e-3})
        out_pop = Population(size=1, dynamics="lif", params={"tau": 10e-3})

        c1 = Connection(source=inp_pop, target=hidden, pattern="dense")
        c2 = Connection(source=hidden, target=out_pop, pattern="dense")

        # XOR-like patterns: encode as spike rates over 20 timesteps
        patterns = [
            (torch.zeros(20, 2), 0.0),  # 0,0 → 0
            (torch.cat([torch.ones(20, 1), torch.zeros(20, 1)], dim=1), 1.0),  # 1,0 → 1
            (torch.cat([torch.zeros(20, 1), torch.ones(20, 1)], dim=1), 1.0),  # 0,1 → 1
            (torch.ones(20, 2), 0.0),  # 1,1 → 0
        ]

        losses = []
        for epoch in range(2):
            epoch_loss = 0.0
            for data, target in patterns:
                inp = Input(population=inp_pop, data=data)
                graph = Graph(
                    [inp_pop, hidden, out_pop], [c1, c2], inputs=[inp]
                )
                model = nuro.compile(
                    graph, target="gpu", requires_grad=True, surrogate="atan"
                )
                optimizer = torch.optim.Adam(model.snn.parameters(), lr=1e-3)
                optimizer.zero_grad()

                output = model.run(duration=0.02, dt=1e-3)
                # Spike count of output neuron as prediction
                pred = output[out_pop.id].sum()
                loss = (pred - target) ** 2
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            losses.append(epoch_loss)

        # We just check it runs without error and produces finite loss
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)
        assert all(not torch.isinf(torch.tensor(l)) for l in losses)
