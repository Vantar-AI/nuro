"""Tests for nuro.plot — visualization tools."""

import numpy as np
import pytest

plt_mod = pytest.importorskip("matplotlib")


class TestSpikePlots:
    def test_spike_raster_returns_axes(self):
        from nuro.plot import spike_raster
        spikes = np.random.randint(0, 2, (100, 10)).astype(float)
        ax = spike_raster(spikes, dt=0.001)
        assert ax is not None

    def test_voltage_traces(self):
        from nuro.plot import voltage_traces
        volts = np.random.randn(100, 5)
        ax = voltage_traces(volts, dt=0.001, neuron_indices=[0, 2])
        assert ax is not None

    def test_firing_rates(self):
        from nuro.plot import firing_rates
        spikes = np.random.randint(0, 2, (200, 8)).astype(float)
        ax = firing_rates(spikes, dt=0.001)
        assert ax is not None

    def test_population_activity(self):
        from nuro.plot import population_activity
        spikes = np.random.randint(0, 2, (100, 10)).astype(float)
        ax = population_activity(spikes, dt=0.001)
        assert ax is not None

    def test_weight_matrix(self):
        from nuro.plot import weight_matrix
        weights = np.random.randn(10, 8)
        ax = weight_matrix(weights)
        assert ax is not None


class TestDashboard:
    def test_experiment_dashboard(self):
        from nuro.plot import experiment_dashboard
        from nuro.recording import Recording

        rec = Recording(dt=0.001)
        rec.extend("spikes", np.random.randint(0, 2, (100, 10)).astype(float))
        rec.extend("voltages", np.random.randn(100, 10))

        fig = experiment_dashboard(rec)
        assert fig is not None


class TestCompare:
    def test_compare_recordings(self):
        from nuro.plot import compare_recordings
        from nuro.recording import Recording

        r1 = Recording(dt=0.001)
        r1.extend("spikes", np.random.randint(0, 2, (50, 5)).astype(float))
        r2 = Recording(dt=0.001)
        r2.extend("spikes", np.random.randint(0, 2, (50, 5)).astype(float))

        ax = compare_recordings({"run1": r1, "run2": r2}, metric="spikes")
        assert ax is not None
