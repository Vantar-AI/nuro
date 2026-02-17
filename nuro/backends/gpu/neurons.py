"""Custom neuron models — Izhikevich and Adaptive Exponential IF (AdEx).

These are implemented as ``nn.Module`` so they integrate with the existing
SpikingJelly-based pipeline (single-step forward, ``.v`` attribute for
voltage recording, compatible with ``functional.reset_net``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IzhikevichNode(nn.Module):
    """Izhikevich neuron model (Euler integration).

    Equations::

        dv/dt = 0.04 v² + 5 v + 140 - u + I
        du/dt = a (b v - u)

        if v >= v_thresh:
            v ← c
            u ← u + d

    Default parameters correspond to **regular spiking** behaviour.

    Parameters
    ----------
    a, b, c, d : float
        Izhikevich parameters.
    v_thresh : float
        Spike threshold (mV).  Default 30.
    dt : float
        Simulation timestep in seconds.  Converted internally to ms.
    """

    PRESETS: dict[str, dict[str, float]] = {
        "regular_spiking": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
        "intrinsically_bursting": {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
        "chattering": {"a": 0.02, "b": 0.2, "c": -50.0, "d": 2.0},
        "fast_spiking": {"a": 0.1, "b": 0.2, "c": -65.0, "d": 2.0},
        "low_threshold_spiking": {"a": 0.02, "b": 0.25, "c": -65.0, "d": 2.0},
    }

    def __init__(
        self,
        size: int,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        v_thresh: float = 30.0,
        dt: float = 1e-3,
    ) -> None:
        super().__init__()
        self.size = size
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.v_thresh = v_thresh
        self.dt_ms = dt * 1000.0  # convert to ms for Izhikevich timescale

        # State
        self.register_buffer("v", torch.full((size,), c))
        self.register_buffer("u", torch.full((size,), b * c))

    def init_state(self, batch_size: int) -> None:
        """Re-initialize state buffers with a batch dimension."""
        device = self.v.device
        self.v = torch.full((batch_size, self.size), self.c, device=device)
        self.u = torch.full((batch_size, self.size), self.b * self.c, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate one timestep.  *x* is synaptic current (arbitrary units,
        scaled to the Izhikevich ~mV range)."""
        # Scale input current: binary spikes need amplification for the
        # Izhikevich voltage range (which operates around -65..+30 mV).
        I = x * 40.0

        # Euler step (0.5 ms sub-steps for stability at dt=1ms)
        steps = max(1, int(self.dt_ms / 0.5))
        for _ in range(steps):
            h = self.dt_ms / steps
            dv = (0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + I) * h
            du = (self.a * (self.b * self.v - self.u)) * h
            self.v = self.v + dv
            self.u = self.u + du

        # Spike detection and reset
        fired = (self.v >= self.v_thresh).float()
        self.v = torch.where(fired.bool(), torch.tensor(self.c, device=self.v.device), self.v)
        self.u = torch.where(fired.bool(), self.u + self.d, self.u)

        return fired

    def reset(self) -> None:
        self.v = torch.full((self.size,), self.c, device=self.v.device)
        self.u = torch.full((self.size,), self.b * self.c, device=self.v.device)


class AdExNode(nn.Module):
    """Adaptive Exponential Integrate-and-Fire (AdEx) neuron.

    Equations::

        C dV/dt = -g_L (V - E_L) + g_L Δ_T exp((V - V_T) / Δ_T) - w + I
        τ_w dw/dt = a (V - E_L) - w

        if V >= V_cutoff:
            V ← V_reset
            w ← w + b

    Parameters
    ----------
    C : float
        Membrane capacitance (pF). Default 281.
    g_L : float
        Leak conductance (nS). Default 30.
    E_L : float
        Leak reversal potential (mV). Default -70.6.
    V_T : float
        Threshold potential (mV). Default -50.4.
    delta_T : float
        Slope factor (mV). Default 2.0.
    a : float
        Subthreshold adaptation (nS). Default 4.0.
    b : float
        Spike-triggered adaptation increment (pA). Default 80.5.
    tau_w : float
        Adaptation time constant (ms). Default 144.
    V_reset : float
        Reset potential (mV). Default -70.6.
    V_cutoff : float
        Spike cutoff potential (mV). Default -40.4.
    dt : float
        Simulation timestep in seconds.
    """

    def __init__(
        self,
        size: int,
        C: float = 281.0,
        g_L: float = 30.0,
        E_L: float = -70.6,
        V_T: float = -50.4,
        delta_T: float = 2.0,
        a: float = 4.0,
        b: float = 80.5,
        tau_w: float = 144.0,
        V_reset: float = -70.6,
        V_cutoff: float = -40.4,
        dt: float = 1e-3,
    ) -> None:
        super().__init__()
        self.size = size
        self.C = C
        self.g_L = g_L
        self.E_L = E_L
        self.V_T = V_T
        self.delta_T = delta_T
        self.a_adapt = a
        self.b_adapt = b
        self.tau_w = tau_w
        self.V_reset = V_reset
        self.V_cutoff = V_cutoff
        self.dt_ms = dt * 1000.0

        self.register_buffer("v", torch.full((size,), E_L))
        self.register_buffer("w", torch.zeros(size))

    def init_state(self, batch_size: int) -> None:
        """Re-initialize state buffers with a batch dimension."""
        device = self.v.device
        self.v = torch.full((batch_size, self.size), self.E_L, device=device)
        self.w = torch.zeros(batch_size, self.size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Integrate one timestep. *x* is synaptic current (scaled internally)."""
        # Scale input: binary spikes → current in pA range
        I = x * 500.0

        h = self.dt_ms  # ms

        # Exponential term (clamped for numerical safety)
        exp_arg = (self.v - self.V_T) / self.delta_T
        exp_arg = torch.clamp(exp_arg, max=20.0)
        exp_term = self.g_L * self.delta_T * torch.exp(exp_arg)

        # Voltage update
        dV = (-self.g_L * (self.v - self.E_L) + exp_term - self.w + I) / self.C * h
        self.v = self.v + dV

        # Adaptation update
        dw = (self.a_adapt * (self.v - self.E_L) - self.w) / self.tau_w * h
        self.w = self.w + dw

        # Spike detection and reset
        fired = (self.v >= self.V_cutoff).float()
        self.v = torch.where(fired.bool(), torch.tensor(self.V_reset, device=self.v.device), self.v)
        self.w = torch.where(fired.bool(), self.w + self.b_adapt, self.w)

        return fired

    def reset(self) -> None:
        self.v = torch.full((self.size,), self.E_L, device=self.v.device)
        self.w = torch.zeros(self.size, device=self.v.device)
