#!/usr/bin/env python3
"""Raw SpikingJelly baseline benchmark for comparison with Nuro.

Measures equivalent LIF → LIF dense networks using SpikingJelly directly,
so we can quantify Nuro's overhead.

Usage:
    python benchmarks/bench_spikingjelly_raw.py
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional, neuron


class RawSNN(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.fc = nn.Linear(size, size, bias=False)
        nn.init.uniform_(self.fc.weight, 0.0, 0.1)
        self.lif = neuron.LIFNode(tau=20.0, step_mode="s")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.fc(x))


def bench_raw(
    neurons: int, duration: float, batch_size: int, dt: float = 1e-3
) -> tuple[float, int]:
    """Return (wall_ms, total_spikes)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RawSNN(neurons).to(device)
    num_steps = int(duration / dt)
    prob = 50.0 * dt

    # Warmup
    functional.reset_net(net)
    for _ in range(10):
        if batch_size > 1:
            x = (torch.rand(batch_size, neurons, device=device) < prob).float()
        else:
            x = (torch.rand(neurons, device=device) < prob).float()
        net(x)
    functional.reset_net(net)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    total = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_steps):
            if batch_size > 1:
                x = (torch.rand(batch_size, neurons, device=device) < prob).float()
            else:
                x = (torch.rand(neurons, device=device) < prob).float()
            spk = net(x)
            total += int(spk.sum().item())

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0

    return wall_ms, total


def main() -> None:
    sizes = [100, 1000, 10000]
    durations = [0.1, 1.0]
    batches = [1, 8, 32, 128]

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Raw SpikingJelly baseline — device: {device}")
    print(f"PyTorch {torch.__version__}")
    print()

    header = f"{'Neurons':>8} {'Dur(s)':>6} {'Batch':>6} {'Time(ms)':>10} {'Spikes':>10}"
    print(header)
    print("-" * len(header))

    for neurons in sizes:
        for dur in durations:
            for batch in batches:
                if not torch.cuda.is_available() and neurons * batch > 50000:
                    continue
                try:
                    wall_ms, total = bench_raw(neurons, dur, batch)
                    print(f"{neurons:>8} {dur:>6.1f} {batch:>6} {wall_ms:>10.1f} {total:>10}")
                except Exception as e:
                    print(f"{neurons:>8} {dur:>6.1f} {batch:>6}  ERROR: {e}")


if __name__ == "__main__":
    main()
