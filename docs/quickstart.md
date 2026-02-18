# Quick Start

Get Nuro running in 5 minutes.

---

## Install

```bash
# From source (recommended — PyPI release coming with v1.0)
git clone https://github.com/Vantar-AI/nuro.git
cd nuro

# GPU backend (training + simulation)
pip install -e ".[gpu,dev]"

# With Loihi backend (requires lava-nc, Python ≤3.10)
pip install -e ".[gpu,loihi,dev]"

# With SpiNNaker 2 backend
pip install -e ".[gpu,spinnaker2,dev]"
```

**Requirements:** Python 3.10+, PyTorch 2.0+, SpikingJelly 0.0.0.14+

---

## Hello Spikes

```python
import nuro

# 1. Define two populations of LIF neurons
input_pop  = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
output_pop = nuro.Population(size=10,  dynamics="lif", params={"tau": 10e-3})

# 2. Connect them (dense all-to-all)
conn = nuro.Connection(source=input_pop, target=output_pop, pattern="dense")

# 3. Build the graph
graph = nuro.Graph([input_pop, output_pop], [conn])

# 4. Compile to GPU backend
model = nuro.compile(graph, target="gpu")

# 5. Run for 1 second
model.run(duration=1.0)

print(f"Total spikes: {model.metrics['total_spikes']}")
```

You just simulated a spiking neural network.

---

## What Just Happened?

1. `Population(size=100, dynamics="lif")` — creates 100 leaky integrate-and-fire neurons
2. `Connection(pattern="dense")` — fully connected synapse matrix (100×10 weights)
3. `Graph(...)` — assembles populations + connections, runs cycle detection
4. `compile(graph, target="gpu")` — lowers to IR, builds a PyTorch `nn.Module`, returns `CompiledModel`
5. `model.run(duration=1.0)` — simulates 1000 timesteps at dt=1ms

---

## Add an Input

Without an `Input`, neurons receive no drive and rarely spike. Let's fix that:

```python
import torch, nuro

pop = nuro.Population(size=50, dynamics="lif")
out = nuro.Population(size=10, dynamics="lif")
conn = nuro.Connection(source=pop, target=out, pattern="dense")

# Option A: Static spike tensor (T, N)
inp = nuro.Input(population=pop, data=torch.rand(1000, 50))

# Option B: Poisson process (rate in Hz)
inp = nuro.Input(population=pop, mode="poisson", rate=100.0)

# Option C: Generator function (called once per timestep)
inp = nuro.Input(population=pop, generator=lambda step: (torch.rand(50) < 0.1).float())

graph = nuro.Graph([pop, out], [conn], inputs=[inp])
model = nuro.compile(graph, target="gpu")
model.run(duration=0.5)
```

---

## Record State

```python
model = nuro.compile(graph, target="gpu")
model.record("spikes",   population=out)
model.record("voltages", population=out)

model.run(duration=1.0)

spikes   = model.get_state("spikes",   population=out)  # (1000, 10)
voltages = model.get_state("voltages", population=out)  # (1000, 10)
```

---

## Change the Backend

Same network, different target:

```python
# GPU (default — training workbench)
model = nuro.compile(graph, target="gpu")

# Intel Loihi 2 (requires lava-nc)
model = nuro.compile(graph, target="loihi")

# SpiNNaker 2 (requires spinnaker2 + Brian2)
model = nuro.compile(graph, target="spinnaker2")
```

Nothing else changes. See [Deployment Guide](deployment.md) for hardware details.

---

## Next Steps

- [Training Guide](training.md) — gradient training with surrogate functions
- [Deployment Guide](deployment.md) — hardware backends
- [API Reference](api-reference.md) — full parameter docs
- [`examples/`](../examples/) — runnable scripts
