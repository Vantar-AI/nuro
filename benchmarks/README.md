# Nuro Benchmarks

Performance benchmarks for Nuro v0.7.0.

## Running

```bash
# Nuro benchmark (all sizes x durations x batch sizes)
python benchmarks/bench_nuro.py

# Save results as JSON
python benchmarks/bench_nuro.py --json results.json

# Raw SpikingJelly baseline
python benchmarks/bench_spikingjelly_raw.py
```

## GPU Simulation Benchmarks

### Configurations

| Parameter | Values |
|-----------|--------|
| Network sizes | 100, 1K, 10K neurons (source -> target, dense) |
| Durations | 0.1s, 1.0s |
| Batch sizes | 1, 8, 32, 128 |
| Neuron model | LIF (tau=20ms -> tau=10ms) |
| Connectivity | Dense, no plasticity |

### What's Measured

- **Wall time (ms)** -- end-to-end simulation time
- **Neuron-steps/s** -- throughput (total neurons x batch x steps / wall time)
- **GPU memory (MB)** -- peak CUDA memory allocation
- **Total spikes** -- sanity check

## Competitive Comparison

| Feature | Nuro v0.7.0 | SpikingJelly | snnTorch | Norse |
|---------|-------------|--------------|----------|-------|
| GPU training | Yes | Yes | Yes | Yes |
| Surrogate gradients | 3 types | 3+ types | 5+ types | 2 types |
| Loihi 2 deploy | Native | No | No | No |
| SpiNNaker 2 deploy | Native | No | No | No |
| Akida deploy | Native | No | No | No |
| NIR support | Import/Export | Export only | Export only | Export only |
| ANN-to-SNN | Yes | No | Yes | No |
| Neuron models | 4 (LIF, IF, Izh, AdEx) | 10+ | 4 | 3 |
| Multi-target compile | Yes (unique) | No | No | No |
| On-chip learning | STDP (Loihi, S2) | No | No | No |
| Dataset loaders | 3 (NMNIST, DVS-CIFAR10, DVS Gesture) | 10+ | 3 | 0 |

### Nuro's Unique Value

**Multi-target hardware compilation** - train once on GPU, deploy to any neuromorphic chip with zero code changes. No other framework offers this.

```python
# Same graph, different targets
gpu_model = nuro.compile(graph, target="gpu", requires_grad=True)
# ... train ...
loihi_model = nuro.compile(graph, target="loihi", weights_from="ckpt.pt")
s2_model = nuro.compile(graph, target="spinnaker2", weights_from="ckpt.pt")
akida_model = nuro.compile(graph, target="akida", weights_from="ckpt.pt")
```

## Hardware Deployment Benchmarks

| Metric | Loihi 2 | SpiNNaker 2 | Akida |
|--------|---------|-------------|-------|
| Weight quantization | 8-bit (auto) | 4-bit (auto) | 4-bit (configurable) |
| Max delay | Native | 7 timesteps | N/A |
| Online learning | STDP | Custom rules | No |
| Power (inference) | ~1W | ~1W | ~300mW |

## Running Hardware Benchmarks

Hardware benchmarks require physical access to neuromorphic chips or cloud access via Vantar Cloud.

```python
# Loihi (requires INRC access)
model = nuro.compile(graph, target="loihi")
model.set_run_config(Loihi2HwCfg())
model.run(duration=1.0)

# SpiNNaker 2 (requires hardware)
model = nuro.compile(graph, target="spinnaker2")
model.set_hardware(SpiNNaker2Chip(eth_ip="192.168.1.1"))
model.run(duration=1.0)

# Akida (requires BrainChip SDK)
model = nuro.compile(graph, target="akida")
model.run(duration=1.0)
```
