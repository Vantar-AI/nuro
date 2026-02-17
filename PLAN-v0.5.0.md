# Nuro v0.5.0 — Intel Loihi Backend

## Context

v0.4.0 shipped surrogate gradients (109 tests, BPTT training with PyTorch optimizers). The GPU backend is now a complete **training** environment. The next step is the reason Nuro exists: **compile the same network to neuromorphic hardware**.

Intel Loihi 2 via the Lava SDK is the first hardware target. Lava provides a simulation mode (`Loihi1SimCfg`) that runs on any machine — no chip access needed for development and testing. When users join INRC and get hardware access, they switch one line (`Loihi2HwCfg`).

**Goal:** `nuro.compile(graph, target="loihi")` produces a runnable model that executes on Lava (simulation or hardware).

---

## Architecture: How Nuro Maps to Lava

```
Nuro IRGraph                    Lava
─────────────                   ────
DynamicsNode (LIF, size=100)  → lava.proc.lif.LIF(shape=(100,), vth=..., dv=..., du=...)
DynamicsNode (IF, size=50)    → lava.proc.lif.LIF(shape=(50,), dv=0, du=0, vth=...)
SynapticEdge (dense)          → lava.proc.dense.Dense(weights=W)
SynapticEdge (sparse)         → lava.proc.dense.Dense(weights=sparse_W)
SynapticEdge (stdp)           → lava.proc.dense.LearningDense(weights=W, learning_rule=STDPLoihi())
Input (static/poisson)        → Custom SpikeGenerator Process
model.run(duration, dt)       → process.run(RunSteps(num_steps), run_cfg=...)
```

### Parameter Mapping: Nuro → Lava

| Nuro Parameter | Lava Parameter | Conversion |
|---------------|----------------|------------|
| `tau` (seconds) | `dv` (decay rate) | `dv = dt / tau` (approximate) |
| `v_thresh` | `vth` | Direct (but scale for fixed-point) |
| Poisson `rate` (Hz) | Bias current | `bias_mant = rate * dt * scale_factor` |
| Connection weights | `Dense.weights` | `nn.Linear.weight.data.numpy()` |

### Key Constraint: Neuron Model Support

Lava natively supports **LIF** only. For Izhikevich and AdEx:
- **Option A:** Custom Lava Process with `PyLoihiProcessModel` (runs in simulation, not on chip)
- **Option B:** Approximate as LIF with adapted parameters (lossy but runs on hardware)
- **Option C:** Use Loihi 2's programmable neuron cores (NcProcessModel) for custom dynamics

**v0.5.0 decision:** Support LIF and IF on Loihi. Izhikevich/AdEx compile to simulation-only mode with a clear warning. Hardware-native custom neurons deferred to v0.6.0.

---

## 5 Steps

### Step 1: Loihi backend scaffold + LIF mapping
**Files:** new `nuro/backends/loihi/`, new `nuro/backends/loihi/backend.py`, new `nuro/backends/loihi/dynamics.py`

- Create `nuro/backends/loihi/__init__.py` with `LoihiBackend`, `LoihiCompiledModel`
- Register `"loihi"` in `nuro/backends/__init__.py` registry
- `LoihiBackend.compile(ir_graph)` → builds Lava process graph:
  - Map each `DynamicsNode` to a `lava.proc.lif.LIF` (or custom Process)
  - Map each `SynapticEdge` to a `lava.proc.dense.Dense`
  - Connect via Lava ports: `neuron.s_out.connect(dense.s_in)`, `dense.a_out.connect(neuron.a_in)`
- Parameter conversion in `dynamics.py`:
  - `build_lava_neuron(node, dt)` → converts tau to dv/du, v_thresh to vth
  - LIF: `LIF(shape=(node.size,), vth=..., dv=dt/tau, du=0.1)`
  - IF: `LIF(shape=(node.size,), vth=..., dv=0, du=0)` (no leak)

### Step 2: LoihiCompiledModel with run() and metrics
**Files:** modify `nuro/backends/loihi/backend.py`, new `nuro/backends/loihi/monitor.py`

- `LoihiCompiledModel.run(duration, dt, batch_size=1)`:
  - Calculate `num_steps = int(duration / dt)`
  - Determine `run_cfg`: `Loihi1SimCfg(select_tag="floating_pt")` by default
  - Allow override: `LoihiCompiledModel.set_run_config(cfg)` for hardware
  - Call `root_process.run(RunSteps(num_steps), run_cfg=cfg)`
  - Collect metrics via Lava Monitor probes
  - `root_process.stop()` after run
- `batch_size > 1`: Not supported on Loihi — raise clear error with message to use GPU backend for batched training
- Metrics: spike counts via Monitor process on output populations
- `reset()`: Stop and recreate Lava processes

### Step 3: Input system (Poisson + static)
**Files:** new `nuro/backends/loihi/inputs.py`

- Custom `SpikeGenerator` Lava Process for input populations:
  - Poisson mode: generate spikes with configurable rate
  - Static mode: replay pre-computed spike tensor
  - Generator mode: not supported on Loihi (raise error with helpful message)
- Wire `SpikeGenerator.s_out` → first Dense layer's `s_in`
- For hardware deployment: static spike data must be pre-loaded

### Step 4: Weight transfer (GPU-trained → Loihi)
**Files:** new `nuro/backends/loihi/transfer.py`

- `transfer_weights(gpu_model, loihi_model)`:
  - Extract `nn.Linear.weight.data` from GPU model's synapses
  - Map to Lava Dense weight matrices
  - Handle weight scaling for fixed-point precision
- Alternative: `nuro.compile(graph, target="loihi", weights_from=gpu_model)`
- This is the key workflow: train on GPU → deploy to Loihi

### Step 5: Tests + example + version bump
**Files:** new `tests/test_loihi_backend.py`, new `examples/deployment/deploy_to_loihi.py`

Tests (require `lava-nc` installed):
- LIF network compiles to Loihi backend
- IF network compiles to Loihi backend
- Izhikevich raises warning (simulation-only)
- Dense connectivity maps correctly
- Sparse connectivity maps correctly
- Static input works
- Poisson input works
- Metrics collected correctly
- Weight transfer from GPU model
- Compile with `target="loihi"` via API
- `batch_size > 1` raises clear error
- Run produces spikes (functional test)

Example `deploy_to_loihi.py`:
- Train a small LIF network on GPU with surrogate gradients
- Transfer weights to Loihi backend
- Run on Loihi simulation
- Compare spike patterns (should be similar)

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default run_cfg | `Loihi1SimCfg(floating_pt)` | Works without hardware, good for development |
| Batch support | Not supported | Loihi runs single inference; use GPU for batch training |
| Izh/AdEx on Loihi | Simulation-only with warning | Custom NcProcessModels need INRC access to test |
| Weight transfer | Explicit function | Users need to control when/how weights move between backends |
| Lava dependency | Optional `[loihi]` extra | Don't require Lava for GPU-only users |
| State recording | Via Lava Monitor | Reuse Lava's built-in probe system |
| STDP on Loihi | Map to LearningDense + STDPLoihi | Native hardware learning |

## Dependencies

```toml
# pyproject.toml addition
[project.optional-dependencies]
loihi = ["lava-nc>=0.9"]
```

## NOT in v0.5.0

- Custom neuron dynamics on Loihi 2 NeuroCores (v0.6.0)
- Multi-chip Loihi deployment
- On-chip training (only inference deployment)
- SpiNNaker backend (v0.6.0)
- Vantar Cloud integration

## Critical Files

| File | Changes |
|------|---------|
| `nuro/backends/loihi/__init__.py` | **New** — LoihiBackend, LoihiCompiledModel exports |
| `nuro/backends/loihi/backend.py` | **New** — LoihiBackend.compile(), LoihiCompiledModel.run() |
| `nuro/backends/loihi/dynamics.py` | **New** — build_lava_neuron(), parameter conversion |
| `nuro/backends/loihi/inputs.py` | **New** — SpikeGenerator Lava Process |
| `nuro/backends/loihi/monitor.py` | **New** — Lava Monitor wrapper for metrics |
| `nuro/backends/loihi/transfer.py` | **New** — Weight transfer GPU → Loihi |
| `nuro/backends/__init__.py` | Register "loihi" in _REGISTRY |
| `pyproject.toml` | Add `loihi` optional dependency |
| `tests/test_loihi_backend.py` | **New** — Loihi backend tests |
| `examples/deployment/deploy_to_loihi.py` | **New** — GPU train → Loihi deploy example |

## Verification

1. `pytest tests/ -v` — all existing tests pass + new Loihi tests
2. `python examples/deployment/deploy_to_loihi.py` — train on GPU, deploy to Loihi sim
3. `python -c "import nuro; print(nuro.__version__)"` → `0.5.0`
4. Existing examples unchanged (GPU default)
5. `nuro.compile(graph, target="loihi")` works without GPU installed

## The Workflow This Enables

```python
import nuro

# Define network (same as always)
inp = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
out = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})
conn = nuro.Connection(source=inp, target=out, pattern="dense")
graph = nuro.Graph([inp, out], [conn], inputs=[...])

# Step 1: Train on GPU
gpu_model = nuro.compile(graph, target="gpu", requires_grad=True)
# ... training loop ...
gpu_model.save("trained.pt")

# Step 2: Deploy to Loihi (one line change)
loihi_model = nuro.compile(graph, target="loihi", weights_from="trained.pt")
loihi_model.run(duration=1.0)
print(loihi_model.metrics)  # Running on neuromorphic silicon!
```
