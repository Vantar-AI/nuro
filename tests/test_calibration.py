"""Tests for nuro.calibration — chip calibration profiles."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nuro.calibration import CalibrationProfile


class TestCalibrationProfile:
    def test_create(self):
        cal = CalibrationProfile(chip_id="board_03", chip_type="dynap-se2")
        assert cal.chip_id == "board_03"
        assert cal.num_neurons == 0

    def test_set_neuron(self):
        cal = CalibrationProfile(chip_id="x", chip_type="y")
        cal.set_neuron(0, threshold=-52.3, tau=18.5e-3)
        cal.set_neuron(1, threshold=-48.1, tau=22.0e-3)
        assert cal.num_neurons == 2
        assert cal.get_neuron(0)["threshold"] == -52.3

    def test_bulk_set(self):
        cal = CalibrationProfile(chip_id="x", chip_type="y")
        thresholds = np.array([-50.0, -51.0, -49.5, -52.0])
        cal.set_neurons_bulk("threshold", thresholds)
        assert cal.num_neurons == 4
        arr = cal.get_param_array("threshold")
        np.testing.assert_allclose(arr, thresholds)

    def test_mismatch_stats(self):
        cal = CalibrationProfile(chip_id="x", chip_type="y")
        cal.set_neurons_bulk("threshold", np.array([-50.0, -52.0, -48.0, -51.0]))
        stats = cal.mismatch_stats("threshold")
        assert "mean" in stats
        assert "std" in stats
        assert "cv" in stats
        assert stats["min"] == -52.0
        assert stats["max"] == -48.0

    def test_mismatch_stats_empty(self):
        cal = CalibrationProfile(chip_id="x", chip_type="y")
        assert cal.mismatch_stats("threshold") == {}


class TestCalibrationHDF5:
    def test_roundtrip(self):
        pytest.importorskip("h5py")

        cal = CalibrationProfile(
            chip_id="board_03",
            chip_type="dynap-se2",
            notes="Morning calibration",
            global_params={"bias_nA": 150, "vdd": 1.8},
        )
        cal.set_neurons_bulk("threshold", np.array([-50.0, -51.0, -49.5]))
        cal.set_neurons_bulk("tau", np.array([20e-3, 18e-3, 22e-3]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "cal.h5")
            cal.save_hdf5(path)
            loaded = CalibrationProfile.load_hdf5(path)

        assert loaded.chip_id == "board_03"
        assert loaded.chip_type == "dynap-se2"
        assert loaded.notes == "Morning calibration"
        assert loaded.num_neurons == 3
        np.testing.assert_allclose(loaded.get_param_array("threshold"), [-50.0, -51.0, -49.5])
        np.testing.assert_allclose(loaded.get_param_array("tau"), [20e-3, 18e-3, 22e-3])


class TestCalibrationWithExperiment:
    def test_experiment_with_calibration(self):
        pytest.importorskip("h5py")
        from nuro.experiment import Experiment

        cal = CalibrationProfile(chip_id="board_03", chip_type="dynap-se2")
        cal.set_neurons_bulk("threshold", np.array([-50.0, -51.0]))

        exp = Experiment(name="cal_test")
        exp.set_calibration(cal)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = exp.save(tmpdir)
            loaded = Experiment.load(path)

        assert loaded._calibration is not None
        assert loaded._calibration.chip_id == "board_03"
        assert loaded._calibration.num_neurons == 2
