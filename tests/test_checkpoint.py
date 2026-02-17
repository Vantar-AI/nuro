"""Tests for checkpointing (save/load)."""

import os
import tempfile

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("spikingjelly")

import nuro
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
    return model, p1, p2


class TestSaveLoad:
    def test_save_creates_file(self):
        model, _, _ = _make_model()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_load_restores_model(self):
        model, p1, p2 = _make_model()
        model.run(duration=0.1, dt=1e-3)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = nuro.load(path)
            loaded.run(duration=0.05, dt=1e-3)
            assert loaded.metrics["num_steps"] == 50
            assert loaded.metrics["total_spikes"] >= 0
        finally:
            os.unlink(path)

    def test_weights_preserved(self):
        model, _, _ = _make_model()
        # Run with STDP to modify weights
        model.run(duration=0.5, dt=1e-3)

        # Get weight snapshot
        original_weights = {}
        for key, syn in model._snn.synapses.items():
            original_weights[key] = syn.weight.detach().cpu().clone()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            loaded = nuro.load(path)

            for key, syn in loaded._snn.synapses.items():
                loaded_w = syn.weight.detach().cpu()
                assert torch.allclose(loaded_w, original_weights[key]), (
                    f"Weights mismatch for synapse {key}"
                )
        finally:
            os.unlink(path)

    def test_version_in_checkpoint(self):
        model, _, _ = _make_model()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            checkpoint = torch.load(path, weights_only=False)
            assert "version" in checkpoint
            assert checkpoint["version"] == nuro.__version__
        finally:
            os.unlink(path)

    def test_version_mismatch_warning(self):
        model, _, _ = _make_model()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            # Tamper with version
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["version"] = "0.0.0"
            torch.save(checkpoint, path)

            with pytest.warns(UserWarning, match="Checkpoint was saved with nuro 0.0.0"):
                nuro.load(path)
        finally:
            os.unlink(path)
