"""Tests for nuro.sweep — parameter sweep for analog experiments."""

import tempfile

import numpy as np
import pytest

from nuro.experiment import Experiment
from nuro.recording import Recording
from nuro.sweep import ParameterSweep


class TestParameterSweep:
    def test_create(self):
        sweep = ParameterSweep(name="bias_sweep", parameter="bias_current", unit="nA")
        assert sweep.num_runs == 0

    def test_add_runs(self):
        sweep = ParameterSweep(name="test", parameter="bias")
        for val in [10, 20, 30]:
            sweep.add_run(val, metrics={"rate": val * 2.0})
        assert sweep.num_runs == 3
        np.testing.assert_array_equal(sweep.values, [10, 20, 30])

    def test_metric_array(self):
        sweep = ParameterSweep(name="test", parameter="bias")
        sweep.add_run(10, metrics={"rate": 20.0})
        sweep.add_run(20, metrics={"rate": 45.0})
        sweep.add_run(30, metrics={"rate": 38.0})
        arr = sweep.metric_array("rate")
        np.testing.assert_array_equal(arr, [20.0, 45.0, 38.0])

    def test_best_max(self):
        sweep = ParameterSweep(name="test", parameter="bias")
        sweep.add_run(10, metrics={"rate": 20.0})
        sweep.add_run(20, metrics={"rate": 45.0})
        sweep.add_run(30, metrics={"rate": 38.0})
        best = sweep.best("rate", mode="max")
        assert best["value"] == 20
        assert best["metric_value"] == 45.0

    def test_best_min(self):
        sweep = ParameterSweep(name="test", parameter="bias")
        sweep.add_run(10, metrics={"loss": 0.5})
        sweep.add_run(20, metrics={"loss": 0.1})
        best = sweep.best("loss", mode="min")
        assert best["value"] == 20

    def test_summary(self):
        sweep = ParameterSweep(name="test", parameter="bias")
        sweep.add_run(10, metrics={"rate": 20.0})
        sweep.add_run(20, metrics={"rate": 45.0})
        s = sweep.summary()
        assert len(s) == 2
        assert s[0]["value"] == 10
        assert s[1]["metrics"]["rate"] == 45.0

    def test_with_experiment(self):
        sweep = ParameterSweep(name="test", parameter="bias")
        exp = Experiment(name="run1")
        exp.log_metric("rate", 42.0)
        sweep.add_run(10, experiment=exp)
        assert sweep._runs[0]["metrics"]["rate"] == 42.0


class TestSweepPlot:
    def test_plot(self):
        pytest.importorskip("matplotlib")
        sweep = ParameterSweep(name="test", parameter="bias", unit="nA")
        sweep.add_run(10, metrics={"rate": 20.0})
        sweep.add_run(20, metrics={"rate": 45.0})
        ax = sweep.plot("rate")
        assert ax is not None


class TestSweepPersistence:
    def test_save_and_load(self):
        pytest.importorskip("h5py")

        sweep = ParameterSweep(name="bias_sweep", parameter="bias_current", unit="nA")
        for val in [10, 20, 30]:
            exp = Experiment(name=f"run_{val}")
            rec = exp.new_recording("main", dt=0.001)
            rec.extend("spikes", np.random.randint(0, 2, (50, 5)).astype(float))
            exp.log_metric("rate", float(val * 2))
            exp.complete()
            sweep.add_run(val, experiment=exp)

        with tempfile.TemporaryDirectory() as tmpdir:
            sweep.save(tmpdir)
            loaded = ParameterSweep.load(f"{tmpdir}/bias_sweep")

        assert loaded.name == "bias_sweep"
        assert loaded.parameter == "bias_current"
        assert loaded.num_runs == 3
        np.testing.assert_array_equal(loaded.values, [10, 20, 30])
