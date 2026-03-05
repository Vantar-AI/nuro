"""Visualization tools for neuromorphic experiments.

All functions accept numpy arrays, return matplotlib axes for composability,
and use lazy imports so matplotlib is only required when plotting.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.axes
    from nuro.recording import Recording


def spike_raster(
    spikes: np.ndarray,
    dt: float = 1e-3,
    ax: Any = None,
    marker_size: float = 2.0,
    color: str = "black",
    alpha: float = 0.8,
    xlabel: str = "Time (ms)",
    ylabel: str = "Neuron index",
    title: str = "Spike Raster",
) -> Any:
    """Scatter plot of spike events.

    Parameters
    ----------
    spikes : ndarray, shape (time, neurons)
        Binary spike array.
    dt : float
        Timestep in seconds.
    """
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    T, N = spikes.shape
    t_ms = np.arange(T) * dt * 1e3

    for neuron_idx in range(N):
        spike_times = t_ms[spikes[:, neuron_idx] > 0.5]
        ax.scatter(
            spike_times, [neuron_idx] * len(spike_times),
            s=marker_size, color=color, alpha=alpha,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, T * dt * 1e3)
    ax.set_ylim(-0.5, N - 0.5)
    return ax


def voltage_traces(
    voltages: np.ndarray,
    dt: float = 1e-3,
    neuron_indices: list[int] | None = None,
    ax: Any = None,
    title: str = "Voltage Traces",
) -> Any:
    """Line plot of membrane voltage over time.

    Parameters
    ----------
    voltages : ndarray, shape (time, neurons)
    neuron_indices : list of int, optional
        Which neurons to plot (default: first 3).
    """
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    T, N = voltages.shape
    t_ms = np.arange(T) * dt * 1e3
    indices = neuron_indices or list(range(min(3, N)))
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]

    for i, idx in enumerate(indices):
        ax.plot(t_ms, voltages[:, idx], color=colors[i % len(colors)],
                linewidth=0.8, label=f"Neuron {idx}", alpha=0.9)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane potential")
    ax.set_title(title)
    ax.set_xlim(0, T * dt * 1e3)
    ax.legend(fontsize=8, loc="upper right")
    return ax


def firing_rates(
    spikes: np.ndarray,
    duration: float | None = None,
    dt: float = 1e-3,
    ax: Any = None,
    title: str = "Firing Rates",
) -> Any:
    """Bar chart of per-neuron firing rates in Hz.

    Parameters
    ----------
    spikes : ndarray, shape (time, neurons)
    duration : float, optional
        Total duration in seconds (default: computed from shape and dt).
    """
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    T, N = spikes.shape
    dur = duration or T * dt
    rates = spikes.sum(axis=0) / dur

    ax.bar(range(N), rates, color="steelblue", alpha=0.8)
    ax.set_xlabel("Neuron index")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title)
    return ax


def population_activity(
    spikes: np.ndarray,
    dt: float = 1e-3,
    ax: Any = None,
    title: str = "Population Activity",
) -> Any:
    """Line plot of total active neurons per timestep.

    Parameters
    ----------
    spikes : ndarray, shape (time, neurons)
    """
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    T = spikes.shape[0]
    t_ms = np.arange(T) * dt * 1e3
    activity = spikes.sum(axis=1)

    ax.plot(t_ms, activity, color="darkorange", linewidth=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Active neurons")
    ax.set_title(title)
    ax.set_xlim(0, T * dt * 1e3)
    return ax


def weight_matrix(
    weights: np.ndarray,
    ax: Any = None,
    cmap: str = "RdBu_r",
    title: str = "Weight Matrix",
) -> Any:
    """Heatmap of a synaptic weight matrix.

    Parameters
    ----------
    weights : ndarray, shape (post, pre)
    """
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    vmax = max(abs(weights.min()), abs(weights.max())) or 1.0
    ax.imshow(weights, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xlabel("Pre-synaptic")
    ax.set_ylabel("Post-synaptic")
    ax.set_title(title)
    return ax


def compare_recordings(
    recordings: dict[str, Recording],
    metric: str = "spikes",
    dt: float = 1e-3,
    ax: Any = None,
    title: str = "Recording Comparison",
) -> Any:
    """Compare population activity across multiple recordings.

    Parameters
    ----------
    recordings : dict mapping label -> Recording
    metric : str
        Probe name to compare.
    """
    plt = _import_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    for label, rec in recordings.items():
        data = rec.get(metric)
        if data.size == 0:
            continue
        t_ms = np.arange(data.shape[0]) * rec.dt * 1e3
        activity = data.sum(axis=1) if data.ndim > 1 else data
        ax.plot(t_ms, activity, linewidth=0.8, label=label, alpha=0.8)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel(f"Total {metric}")
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def experiment_dashboard(
    recording: Recording,
    save_path: str | None = None,
) -> Any:
    """4-panel composite: raster, voltages, rates, population activity.

    Uses whatever probes are available in the recording. Expects "spikes"
    and optionally "voltages" probe names.
    """
    plt = _import_matplotlib()
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    spikes_data = recording.get("spikes")
    has_spikes = spikes_data.size > 0
    volts_data = recording.get("voltages")
    has_volts = volts_data.size > 0

    if has_spikes:
        spike_raster(spikes_data, dt=recording.dt, ax=fig.add_subplot(gs[0, 0]))
        firing_rates(spikes_data, dt=recording.dt, ax=fig.add_subplot(gs[1, 0]))
        population_activity(spikes_data, dt=recording.dt, ax=fig.add_subplot(gs[1, 1]))

    if has_volts:
        voltage_traces(volts_data, dt=recording.dt, ax=fig.add_subplot(gs[0, 1]))

    fig.suptitle("Nuro Experiment Dashboard", fontsize=13, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend for safety
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install nuro[devtools]"
        )
