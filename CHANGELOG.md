# Changelog

All notable changes to Nuro are documented here.

## [0.2.0] - 2026-02-17

### Added
- **User Input System** — `nuro.Input` class supporting static tensors, generator functions, and configurable Poisson rates
- **State Recording** — `model.record()` and `model.get_state()` for voltages, spikes, and weight snapshots
- **Izhikevich neurons** — Full Izhikevich model with presets (regular spiking, fast spiking, chattering, bursting, LTS)
- **AdEx neurons** — Adaptive Exponential Integrate-and-Fire model
- **Recurrent graph support** — Automatic cycle detection, topological sort for DAGs, Jacobi iteration for cyclic graphs
- **Checkpointing** — `model.save()` and `nuro.load()` for saving/restoring trained networks
- **`Graph.is_cyclic`** property for cycle detection
- 4 new examples: custom input, state recording, Izhikevich networks, recurrent networks
- 40 new tests (77 total)

### Changed
- `Graph` now accepts optional `inputs` kwarg
- `SUPPORTED_DYNAMICS` expanded to include `"izhikevich"` and `"adex"`
- GPU backend uses topological sort for DAG execution order (was insertion order)
- Source population detection now includes explicitly-driven populations in cyclic graphs

## [0.1.0] - 2026-02-17

### Added
- Core API: `Population`, `Connection`, `Graph`, `compile()`
- Intermediate Representation: `DynamicsNode`, `SynapticEdge`, `IRGraph`
- GPU backend via SpikingJelly (LIF, IF neurons)
- Dense and random sparse connectivity
- Trace-based STDP plasticity
- 37 tests
- Hello Spikes example
