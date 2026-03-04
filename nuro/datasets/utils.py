"""Dataset utilities — downloading and event-to-tensor conversion."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def get_cache_dir(root: str | None = None) -> Path:
    """Get or create the dataset cache directory."""
    if root is not None:
        path = Path(root)
    else:
        path = Path.home() / ".cache" / "nuro" / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def events_to_spike_tensor(
    events: np.ndarray,
    num_neurons: int,
    duration_ms: float,
    dt: float = 1e-3,
) -> np.ndarray:
    """Convert event stream to dense spike tensor.

    Parameters
    ----------
    events : np.ndarray
        Structured array with fields 'x', 'y', 't', 'p' (polarity).
        Or array with columns [x, y, t, p].
    num_neurons : int
        Total number of neurons (e.g. 34*34*2 for DVS).
    duration_ms : float
        Duration to bin events into, in milliseconds.
    dt : float
        Timestep in seconds. Default 1ms.

    Returns
    -------
    np.ndarray
        Shape ``(num_steps, num_neurons)`` binary spike tensor.
    """
    dt_ms = dt * 1000
    num_steps = int(duration_ms / dt_ms)
    tensor = np.zeros((num_steps, num_neurons), dtype=np.float32)

    if len(events) == 0:
        return tensor

    # Handle both structured and plain arrays
    if hasattr(events, "dtype") and events.dtype.names:
        times = events["t"]
        addresses = events["x"] + events["y"] * 34  # Simple linearization
    else:
        times = events[:, 2]
        addresses = (events[:, 0] + events[:, 1] * 34).astype(int)

    # Normalize timestamps to bin indices
    if len(times) > 0:
        t_min = times.min()
        t_range = times.max() - t_min
        if t_range > 0:
            bin_indices = ((times - t_min) / t_range * (num_steps - 1)).astype(int)
        else:
            bin_indices = np.zeros_like(times, dtype=int)

        # Clip addresses to valid range
        addresses = np.clip(addresses, 0, num_neurons - 1)
        bin_indices = np.clip(bin_indices, 0, num_steps - 1)

        tensor[bin_indices, addresses] = 1.0

    return tensor


def download_file(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    """Download a file with progress reporting.

    Parameters
    ----------
    url : str
        URL to download.
    dest : Path
        Destination file path.
    expected_md5 : str, optional
        Expected MD5 hash for verification.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    import urllib.request

    if dest.exists():
        if expected_md5 is not None:
            actual = hashlib.md5(dest.read_bytes()).hexdigest()
            if actual == expected_md5:
                return dest
        else:
            return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} → {dest}")
    urllib.request.urlretrieve(url, dest)

    if expected_md5 is not None:
        actual = hashlib.md5(dest.read_bytes()).hexdigest()
        if actual != expected_md5:
            dest.unlink()
            raise RuntimeError(
                f"MD5 mismatch for {dest.name}: {actual} != {expected_md5}"
            )

    return dest
