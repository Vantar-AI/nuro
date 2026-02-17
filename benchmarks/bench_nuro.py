#!/usr/bin/env python3
"""Nuro performance benchmarks.

Measures wall time, throughput, and GPU memory for various network sizes,
durations, and batch sizes.

Usage:
    python benchmarks/bench_nuro.py
    python benchmarks/bench_nuro.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass

import torch

import nuro
from nuro.backends.gpu.backend import GPUBackend
from nuro.ir import IRGraph


@dataclass
class BenchResult:
    neurons: int
    duration_s: float
    batch_size: int
    wall_ms: float
    spikes_per_sec: float
    gpu_mem_mb: float
    total_spikes: int
    num_steps: int


def bench_config(
    neurons: int,
    duration: float,
    batch_size: int,
    dt: float = 1e-3,
) -> BenchResult:
    """Run a single benchmark configuration."""
    src = nuro.Population(size=neurons, dynamics="lif", params={"tau": 20e-3})
    tgt = nuro.Population(size=neurons, dynamics="lif", params={"tau": 10e-3})
    conn = nuro.Connection(source=src, target=tgt, pattern="dense")
    graph = nuro.Graph([src, tgt], [conn])
    ir = IRGraph.from_api_graph(graph)

    model = GPUBackend().compile(ir)

    # Warmup
    model.run(duration=0.01, dt=dt, batch_size=batch_size)
    model.reset()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    model.run(duration=duration, dt=dt, batch_size=batch_size)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0

    total_spikes = model.metrics["total_spikes"]
    num_steps = model.metrics["num_steps"]

    # Throughput: spikes that could be processed per second of wall time
    sim_neurons = neurons * 2 * batch_size  # src + tgt, across batch
    spikes_per_sec = sim_neurons * num_steps / (wall_ms / 1000.0) if wall_ms > 0 else 0

    gpu_mem_mb = 0.0
    if torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return BenchResult(
        neurons=neurons,
        duration_s=duration,
        batch_size=batch_size,
        wall_ms=round(wall_ms, 2),
        spikes_per_sec=round(spikes_per_sec),
        gpu_mem_mb=round(gpu_mem_mb, 1),
        total_spikes=total_spikes,
        num_steps=num_steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Nuro benchmarks")
    parser.add_argument("--json", type=str, help="Output JSON file path")
    args = parser.parse_args()

    sizes = [100, 1000, 10000]
    durations = [0.1, 1.0]
    batches = [1, 8, 32, 128]

    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Nuro v{nuro.__version__} benchmark — device: {device}")
    print(f"PyTorch {torch.__version__}")
    print()

    header = f"{'Neurons':>8} {'Dur(s)':>6} {'Batch':>6} {'Time(ms)':>10} {'Neuron-steps/s':>16} {'GPU MB':>8} {'Spikes':>10}"
    print(header)
    print("-" * len(header))

    results: list[dict] = []

    for neurons in sizes:
        for dur in durations:
            for batch in batches:
                # Skip very large configs on CPU to keep bench reasonable
                if not torch.cuda.is_available() and neurons * batch > 50000:
                    continue

                try:
                    r = bench_config(neurons, dur, batch)
                    print(
                        f"{r.neurons:>8} {r.duration_s:>6.1f} {r.batch_size:>6} "
                        f"{r.wall_ms:>10.1f} {r.spikes_per_sec:>16,.0f} "
                        f"{r.gpu_mem_mb:>8.1f} {r.total_spikes:>10}"
                    )
                    results.append(asdict(r))
                except Exception as e:
                    print(f"{neurons:>8} {dur:>6.1f} {batch:>6}  ERROR: {e}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.json}")


if __name__ == "__main__":
    main()
