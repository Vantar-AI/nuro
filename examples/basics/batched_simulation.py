#!/usr/bin/env python3
"""Batched Simulation — run 32 parallel trials and compare variance.

Demonstrates Nuro's batch support: a single model.run() call simulates
multiple independent trials with different random Poisson inputs, producing
different spike patterns per trial.

Usage:
    python examples/basics/batched_simulation.py
"""

import torch

import nuro

# --- Network definition (same as single-trial) ---
sensory = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
motor = nuro.Population(size=20, dynamics="lif", params={"tau": 10e-3})
conn = nuro.Connection(source=sensory, target=motor, pattern="dense")
graph = nuro.Graph([sensory, motor], [conn])

# --- Compile and run 32 trials in parallel ---
model = nuro.compile(graph, target="gpu")
model.record("spikes", population=motor)

BATCH_SIZE = 32
model.run(duration=1.0, batch_size=BATCH_SIZE)

# --- Analyze results ---
spikes = model.get_state("spikes", population=motor)
print(f"Recorded spike tensor shape: {spikes.shape}")
# Expected: (1000, 32, 20) = (steps, batch, pop_size)

# Spike count per trial
per_trial = spikes.sum(dim=(0, 2))  # (32,)
print(f"\nSpike counts across {BATCH_SIZE} trials:")
print(f"  Mean:  {per_trial.mean().item():.1f}")
print(f"  Std:   {per_trial.std().item():.1f}")
print(f"  Min:   {per_trial.min().item():.0f}")
print(f"  Max:   {per_trial.max().item():.0f}")

# Compare: running without batch (backward compatible)
model2 = nuro.compile(graph, target="gpu")
model2.record("spikes", population=motor)
model2.run(duration=1.0)  # batch_size=1 by default
spikes_single = model2.get_state("spikes", population=motor)
print(f"\nSingle trial spike tensor shape: {spikes_single.shape}")
# Expected: (1000, 20) = (steps, pop_size)

print(f"\nTotal spikes: {model2.metrics['total_spikes']}")
print(f"Batch size:   {model2.metrics['batch_size']}")
