"""Tests for nuro.experiment — Experiment tracking."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from nuro.experiment import Experiment, HardwareConfig
from nuro.recording import Recording


class TestExperimentBasic:
    def test_create_experiment(self):
        exp = Experiment(name="test", project="my_project", tags=["snn"])
        assert exp.name == "test"
        assert exp.project == "my_project"
        assert exp.status == "running"
        assert len(exp.id) == 12

    def test_set_hardware(self):
        exp = Experiment(name="test")
        exp.set_hardware("loihi2", chip_count=2)
        assert exp._hardware.platform == "loihi2"
        assert exp._hardware.params["chip_count"] == 2

    def test_log_metrics(self):
        exp = Experiment(name="test")
        exp.log_metric("accuracy", 0.95)
        exp.log_metrics({"loss": 0.1, "spikes": 1000})
        assert exp.metrics["accuracy"] == 0.95
        assert exp.metrics["loss"] == 0.1

    def test_set_params(self):
        exp = Experiment(name="test")
        exp.set_params(lr=0.01, epochs=10)
        assert exp._params["lr"] == 0.01

    def test_complete(self):
        exp = Experiment(name="test")
        exp.complete()
        assert exp.status == "completed"


class TestRecordings:
    def test_new_recording(self):
        exp = Experiment(name="test")
        rec = exp.new_recording("main", dt=0.001)
        assert isinstance(rec, Recording)
        assert "main" in exp.recordings

    def test_add_existing_recording(self):
        exp = Experiment(name="test")
        rec = Recording(dt=0.001)
        rec.append("spikes", np.array([1, 0, 1]))
        exp.add_recording("imported", rec)
        assert exp.get_recording("imported") is rec

    def test_get_recording_raises(self):
        exp = Experiment(name="test")
        with pytest.raises(KeyError):
            exp.get_recording("nonexistent")


class TestPersistence:
    def test_save_and_load_roundtrip(self):
        pytest.importorskip("h5py")

        exp = Experiment(
            name="roundtrip_test",
            project="tests",
            description="Testing save/load",
            tags=["test", "ci"],
        )
        exp.set_hardware("gpu")
        exp.set_params(lr=0.001, batch_size=32)
        exp.log_metrics({"accuracy": 0.92, "total_spikes": 5000})

        rec = exp.new_recording("spikes", dt=0.001)
        rec.add_probe("spikes")
        rec.extend("spikes", np.random.randint(0, 2, (100, 10)))

        exp.complete()

        with tempfile.TemporaryDirectory() as tmpdir:
            saved_path = exp.save(tmpdir)
            assert (saved_path / "experiment.json").exists()
            assert (saved_path / "recording_spikes.h5").exists()

            loaded = Experiment.load(saved_path)

        assert loaded.id == exp.id
        assert loaded.name == "roundtrip_test"
        assert loaded.status == "completed"
        assert loaded.metrics["accuracy"] == 0.92
        assert loaded._hardware.platform == "gpu"
        assert loaded.get_recording("spikes").num_steps == 100

    def test_save_without_recordings(self):
        exp = Experiment(name="empty")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exp.save(tmpdir)
            loaded = Experiment.load(path)
            assert loaded.name == "empty"
            assert len(loaded.recordings) == 0


class TestHardwareConfig:
    def test_roundtrip(self):
        hw = HardwareConfig(platform="loihi2", chip_id="board_01", params={"cores": 128})
        d = hw.to_dict()
        hw2 = HardwareConfig.from_dict(d)
        assert hw2.platform == "loihi2"
        assert hw2.chip_id == "board_01"
        assert hw2.params["cores"] == 128
