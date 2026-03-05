"""AER (Address-Event Representation) binary format adapter.

Parses raw AER data files produced by analog neuromorphic hardware.
Supports jAER's .aedat 2.0 format, generic binary, and pre-parsed arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from nuro.recording import Recording


def from_aedat(
    path: str | Path,
    num_neurons: int,
    dt: float = 1e-3,
    addr_mask: int = 0x0000FFFF,
    addr_shift: int = 0,
) -> Recording:
    """Parse a jAER .aedat 2.0 file into a Recording.

    The aedat 2.0 format stores events as repeating 8-byte records:
    4 bytes address (big-endian int32) + 4 bytes timestamp (big-endian int32, microseconds).

    Parameters
    ----------
    path : path
        Path to .aedat file.
    num_neurons : int
        Total neuron count for binning.
    dt : float
        Bin width in seconds.
    addr_mask : int
        Bitmask to extract neuron ID from address word.
    addr_shift : int
        Right-shift applied after masking.
    """
    raw = Path(path).read_bytes()

    # Skip text header lines (start with '#' or '%')
    offset = 0
    while offset < len(raw) and raw[offset:offset + 1] in (b"#", b"%"):
        offset = raw.index(b"\n", offset) + 1

    data = raw[offset:]
    if len(data) < 8:
        rec = Recording(dt=dt, source="aedat")
        rec.add_probe("spikes")
        return rec

    # Parse as big-endian int32 pairs
    n_events = len(data) // 8
    buf = np.frombuffer(data[: n_events * 8], dtype=">i4").reshape(-1, 2)
    addresses = buf[:, 0].astype(np.int64)
    timestamps_us = buf[:, 1].astype(np.int64)

    neuron_ids = (addresses & addr_mask) >> addr_shift
    timestamps_s = (timestamps_us - timestamps_us[0]) * 1e-6

    return from_aer_events(neuron_ids, timestamps_s, num_neurons, dt)


def from_aer_binary(
    path: str | Path,
    num_neurons: int,
    dt: float = 1e-3,
    addr_bytes: int = 4,
    ts_bytes: int = 4,
    big_endian: bool = True,
    ts_scale: float = 1e-6,
    addr_mask: int = 0xFFFFFFFF,
    addr_shift: int = 0,
) -> Recording:
    """Parse a generic AER binary file.

    Expects repeating records of (address, timestamp) with configurable
    byte widths and endianness.

    Parameters
    ----------
    ts_scale : float
        Multiply raw timestamps by this to get seconds (default 1e-6 for microseconds).
    """
    raw = Path(path).read_bytes()
    record_size = addr_bytes + ts_bytes
    n_events = len(raw) // record_size

    if n_events == 0:
        rec = Recording(dt=dt, source="aer-binary")
        rec.add_probe("spikes")
        return rec

    endian = ">" if big_endian else "<"
    addr_dtype = f"{endian}u{addr_bytes}"
    ts_dtype = f"{endian}u{ts_bytes}"

    addresses = np.zeros(n_events, dtype=np.int64)
    timestamps = np.zeros(n_events, dtype=np.float64)

    for i in range(n_events):
        off = i * record_size
        addresses[i] = int.from_bytes(raw[off : off + addr_bytes], "big" if big_endian else "little")
        timestamps[i] = int.from_bytes(raw[off + addr_bytes : off + record_size], "big" if big_endian else "little")

    neuron_ids = (addresses & addr_mask) >> addr_shift
    timestamps_s = (timestamps - timestamps[0]) * ts_scale

    return from_aer_events(neuron_ids, timestamps_s, num_neurons, dt)


def from_aer_events(
    neuron_ids: np.ndarray,
    timestamps_s: np.ndarray,
    num_neurons: int,
    dt: float = 1e-3,
    duration: float | None = None,
) -> Recording:
    """Create a Recording from pre-parsed AER address and timestamp arrays.

    Parameters
    ----------
    neuron_ids : ndarray of int
        Neuron ID per event (extracted from AER address).
    timestamps_s : ndarray of float
        Event timestamps in seconds.
    num_neurons : int
        Total number of neurons for binning.
    dt : float
        Bin width in seconds.
    duration : float, optional
        Total duration. If None, inferred from last event.
    """
    rec = Recording(dt=dt, source="aer")
    rec.add_probe("spikes")

    if len(neuron_ids) == 0:
        return rec

    max_t = timestamps_s.max()
    dur = duration or (max_t + dt)
    num_steps = int(dur / dt)

    spikes = np.zeros((num_steps, num_neurons), dtype=np.float32)

    steps = (timestamps_s / dt).astype(np.int64)
    valid = (steps >= 0) & (steps < num_steps) & (neuron_ids >= 0) & (neuron_ids < num_neurons)
    spikes[steps[valid], neuron_ids[valid].astype(np.int64)] = 1.0

    rec.extend("spikes", spikes)
    return rec
