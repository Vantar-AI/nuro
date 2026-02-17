# Nuro Benchmarks

Performance benchmarks comparing Nuro against raw SpikingJelly.

## Running

```bash
# Nuro benchmark (all sizes × durations × batch sizes)
python benchmarks/bench_nuro.py

# Save results as JSON
python benchmarks/bench_nuro.py --json results.json

# Raw SpikingJelly baseline
python benchmarks/bench_spikingjelly_raw.py
```

## Configurations

| Parameter | Values |
|-----------|--------|
| Network sizes | 100, 1K, 10K neurons (source → target, dense) |
| Durations | 0.1s, 1.0s |
| Batch sizes | 1, 8, 32, 128 |
| Neuron model | LIF (tau=20ms → tau=10ms) |
| Connectivity | Dense, no plasticity |

## What's Measured

- **Wall time (ms)** — end-to-end simulation time
- **Neuron-steps/s** — throughput (total neurons × batch × steps / wall time)
- **GPU memory (MB)** — peak CUDA memory allocation
- **Total spikes** — sanity check
