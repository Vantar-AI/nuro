"""File-based adapters — import recordings from CSV, HDF5, numpy, and NIR."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nuro.recording import Recording


def from_csv(
    path: str | Path,
    probe_name: str = "data",
    dt: float = 1e-3,
    delimiter: str = ",",
    skip_header: int = 0,
) -> Recording:
    """Load a Recording from a CSV file.

    Expects a 2D array where rows = timesteps, columns = channels.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.
    probe_name : str
        Name for the probe (default "data").
    dt : float
        Timestep in seconds.
    delimiter : str
        Column delimiter.
    skip_header : int
        Number of header rows to skip.
    """
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skip_header)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    rec = Recording(dt=dt, source="csv")
    rec.add_probe(probe_name)
    rec.extend(probe_name, data)
    return rec


def from_hdf5(path: str | Path, dt: float = 1e-3) -> Recording:
    """Load a Recording from an HDF5 file.

    If the file was saved by ``Recording.save_hdf5()``, use
    ``Recording.load_hdf5()`` instead. This function handles
    generic HDF5 files by importing each top-level dataset as a probe.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py is required: pip install nuro[devtools]")

    rec = Recording(dt=dt, source="hdf5")

    with h5py.File(path, "r") as f:
        for key in f:
            if isinstance(f[key], h5py.Dataset):
                data = np.array(f[key])
                rec.add_probe(key)
                rec.extend(key, data)

    return rec


def from_numpy(
    data: np.ndarray,
    probe_name: str = "data",
    dt: float = 1e-3,
    source: str = "numpy",
) -> Recording:
    """Wrap a numpy array as a Recording.

    Parameters
    ----------
    data : ndarray
        Array where first axis = time.
    probe_name : str
        Name for the probe.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    rec = Recording(dt=dt, source=source)
    rec.add_probe(probe_name)
    rec.extend(probe_name, data)
    return rec


def from_nir_file(path: str | Path, dt: float = 1e-3) -> Recording:
    """Load a NIR file and extract any recorded data.

    NIR files primarily describe network architecture, but some
    include recorded spike/voltage data as node attributes.
    """
    try:
        import nir as nir_mod
    except ImportError:
        raise ImportError("nir is required: pip install nuro[nir]")

    nir_graph = nir_mod.read(str(path))
    rec = Recording(dt=dt, source="nir")

    # Extract any data arrays attached to nodes
    for node_key, node in nir_graph.nodes.items():
        if hasattr(node, "output") and node.output is not None:
            data = np.asarray(node.output)
            if data.size > 0:
                rec.add_probe("output", target_id=node_key)
                rec.extend("output", data, target_id=node_key)
        if hasattr(node, "spikes") and node.spikes is not None:
            data = np.asarray(node.spikes)
            if data.size > 0:
                rec.add_probe("spikes", target_id=node_key)
                rec.extend("spikes", data, target_id=node_key)

    return rec
