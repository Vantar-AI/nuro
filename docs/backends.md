# Backend Comparison

Nuro compiles the same network definition to multiple backends. This document explains each backend's purpose, capabilities, and tradeoffs.

---

## At a Glance

| Backend | Target | Training | Deployment | Hardware |
|---------|--------|----------|-----------|---------|
| `gpu` | RTX/A100/M-series | ✓ Surrogate gradients | Simulation only | GPU |
| `loihi` | Intel Loihi 2 | ✗ | ✓ Real silicon | INRC (free) |
| `spinnaker2` | SpiNNaker 2 | ✗ | ✓ Real silicon | SpiNNcloud |
| `cloud` | Any | ✗ | ✓ Remote deploy | Vantar Cloud (beta) |

**The workflow:** GPU for training, neuromorphic silicon for deployment.

---

## GPU Backend

**Module:** `nuro.backends.gpu`
**Library:** SpikingJelly + PyTorch
**Install:** `pip install -e ".[gpu]"`

The training workbench. Every researcher starts here.

**What it does:**
- Compiles your network to a `torch.nn.Module` (`NuroSNN`)
- Implements surrogate gradients for backpropagation through spikes
- Supports all neuron models (LIF, IF, Izhikevich, AdEx)
- Batched simulation for 32–128× throughput scaling
- State recording (voltages, spikes, weights)
- Checkpointing (save/load)

**When to use:** Always. GPU is your development and training environment.

**Compile options:**
```python
model = nuro.compile(
    graph,
    target="gpu",
    requires_grad=True,    # Enable surrogate gradients for training
    surrogate="atan",      # "atan" | "sigmoid" | "triangular"
)
```

---

## Loihi Backend

**Module:** `nuro.backends.loihi`
**Library:** lava-nc (Intel Lava framework)
**Install:** `pip install -e ".[loihi]"` (Python ≤3.10)

Intel Loihi 2 deployment. 1000× more energy efficient than GPU inference.

**What it does:**
- Compiles to a Lava Process graph (`lava.proc`)
- Runs on Loihi 2 Sim (default) or real Loihi 2 hardware (with INRC access)
- Transfers weights from trained GPU checkpoint
- Spike monitoring via Lava probes

**Neuron models on Loihi:**
- LIF, IF: native (Lava `LIF` Process)
- Izhikevich, AdEx: simulation-only in v0.5; NcProcess support in v0.6

**When to use:** Final deployment after GPU training. Edge inference, power-constrained applications.

**Compile options:**
```python
model = nuro.compile(
    graph,
    target="loihi",
    weights_from="trained.pt",   # Load GPU-trained weights
)
```

**Switch to hardware:**
```python
from lava.magma.core.run_configs import Loihi2HwCfg
model.set_run_config(Loihi2HwCfg())
```

---

## SpiNNaker 2 Backend

**Module:** `nuro.backends.spinnaker2`
**Library:** py-spinnaker2 + Brian2
**Install:** `pip install -e ".[spinnaker2]"`

Manchester's SpiNNaker 2 massively parallel neuromorphic system.

**What it does:**
- Compiles to a py-spinnaker2 `Network`
- Brian2 as simulation fallback (no hardware needed)
- Weight transfer from GPU checkpoint
- Spike recording via `get_spikes()`

**When to use:** Research on large-scale networks, comparison with Loihi, SpiNNcloud access.

**Compile options:**
```python
model = nuro.compile(
    graph,
    target="spinnaker2",
    weights_from="trained.pt",
)
```

**Switch to hardware:**
```python
model.set_hardware(True)
```

---

## Cloud Backend (v0.7.0)

**Module:** `nuro.backends.cloud`
**Library:** requests (HTTP client)
**Install:** Included in core (no extra deps)

Submit to Vantar Cloud API — we compile and run on real hardware for you.

**What it does:**
- Serializes your IRGraph to JSON
- POSTs to Vantar Cloud API (`https://api.vantar.xyz`)
- Polls for compilation/execution results
- Returns same metrics interface as local backends

**When to use:** You don't have INRC or SpiNNcloud access. You want to evaluate neuromorphic performance before committing to hardware access programs.

**Compile options:**
```python
model = nuro.compile(
    graph,
    target="cloud",
    hardware="loihi",          # Target hardware on cloud
    weights_from="trained.pt",
    api_key="vt_...",          # Or set VANTAR_API_KEY env var
)
```

[Join the Vantar Cloud waitlist →](https://vantar.xyz)

---

## Setup Checklist

### GPU
```bash
pip install torch>=2.0 spikingjelly>=0.0.0.0.14
pip install -e ".[gpu]"
python -c "import nuro; print(nuro.__version__)"
```

### Loihi
```bash
# Requires Python 3.10 (lava-nc constraint)
python3.10 -m venv .venv-loihi && source .venv-loihi/bin/activate
pip install lava-nc>=0.9
pip install -e ".[loihi]"
```

### SpiNNaker 2
```bash
pip install spinnaker2>=0.5 brian2
pip install -e ".[spinnaker2]"
```

### Cloud
```bash
pip install -e "."   # No extra deps
export VANTAR_API_KEY=vt_your_key_here
```
