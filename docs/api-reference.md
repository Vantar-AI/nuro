# API Reference

All public classes and functions in the `nuro` namespace.

---

## `nuro.Population`

A group of neurons with shared dynamics.

```python
nuro.Population(
    size: int,
    dynamics: str = "lif",
    params: dict = None,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `size` | `int` | Number of neurons |
| `dynamics` | `str` | Neuron model: `"lif"`, `"if"`, `"izhikevich"`, `"adex"` |
| `params` | `dict` | Model-specific parameters (see below) |

**LIF parameters:**

| Key | Default | Description |
|-----|---------|-------------|
| `tau` | `20e-3` | Membrane time constant (seconds) |
| `v_thresh` | `-50.0` | Spike threshold (mV) |
| `v_reset` | `-70.0` | Reset potential (mV) |
| `v_rest` | `-65.0` | Resting potential (mV) |

**Izhikevich parameters:**

| Key | Default | Description |
|-----|---------|-------------|
| `preset` | `None` | Preset name (see below) or None for custom |
| `a` | `0.02` | Recovery time scale |
| `b` | `0.2` | Recovery sensitivity |
| `c` | `-65.0` | After-spike reset voltage |
| `d` | `8.0` | After-spike recovery increment |

**Izhikevich presets:** `"regular_spiking"`, `"intrinsically_bursting"`, `"chattering"`, `"fast_spiking"`, `"low_threshold_spiking"`

**Examples:**
```python
lif  = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
izh  = nuro.Population(size=200, dynamics="izhikevich", params={"preset": "fast_spiking"})
adex = nuro.Population(size=50,  dynamics="adex")
```

---

## `nuro.Connection`

Synaptic projection between two populations.

```python
nuro.Connection(
    source: Population,
    target: Population,
    pattern: str = "dense",
    plasticity: str = None,
    weight_scale: float = 1.0,
    sparsity: float = 0.1,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `Population` | Pre-synaptic population |
| `target` | `Population` | Post-synaptic population |
| `pattern` | `str` | `"dense"` (all-to-all) or `"random_sparse"` |
| `plasticity` | `str` | `"stdp"` or `None` |
| `weight_scale` | `float` | Weight initialization scale |
| `sparsity` | `float` | Connection density for `"random_sparse"` (0–1) |

---

## `nuro.Input`

External input to a population.

```python
nuro.Input(
    population: Population,
    data: Tensor = None,
    generator: Callable = None,
    mode: str = None,
    rate: float = 100.0,
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `population` | `Population` | Target population |
| `data` | `Tensor(T, N)` | Static spike tensor; `T` steps, `N` neurons |
| `generator` | `Callable(step) → Tensor(N)` | Called once per timestep |
| `mode` | `str` | `"poisson"` for Poisson spike generation |
| `rate` | `float` | Poisson rate in Hz (used when `mode="poisson"`) |

Exactly one of `data`, `generator`, or `mode` should be provided.

---

## `nuro.Graph`

Container for a complete network definition.

```python
nuro.Graph(
    populations: list[Population],
    connections: list[Connection],
    inputs: list[Input] = None,
)
```

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `is_cyclic` | `bool` | True if graph contains recurrent connections |
| `populations` | `list` | All population nodes |
| `connections` | `list` | All connection edges |

---

## `nuro.compile`

Compile a graph to a runnable model.

```python
nuro.compile(
    graph: Graph,
    target: str = "gpu",
    **kwargs,
) → CompiledModel
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `graph` | `Graph` | Network to compile |
| `target` | `str` | Backend: `"gpu"`, `"loihi"`, `"spinnaker2"`, `"cloud"` |
| `requires_grad` | `bool` | Enable surrogate gradients (GPU only) |
| `surrogate` | `str` | Gradient function: `"atan"`, `"sigmoid"`, `"triangular"` |
| `weights_from` | `str` | Path to GPU checkpoint for weight transfer |
| `hardware` | `str` | Target hardware for cloud backend: `"loihi"`, `"spinnaker2"` |
| `api_key` | `str` | Vantar Cloud API key (or `VANTAR_API_KEY` env var) |

---

## `CompiledModel`

Returned by `nuro.compile()`. Abstract interface implemented by each backend.

### `model.run()`

```python
model.run(
    duration: float,
    dt: float = 1e-3,
    batch_size: int = 1,
) → None | dict[str, Tensor]
```

- `duration`: simulation time in seconds
- `dt`: timestep in seconds (default 1 ms)
- `batch_size`: parallel trials (GPU only)
- **Returns:** `None` by default; `dict[pop_id → Tensor]` when `requires_grad=True`

### `model.reset()`

Reset all membrane potentials and internal state to initial conditions. Call between training epochs.

### `model.record()`

```python
model.record(
    name: str,
    population: Population = None,
    connection: Connection = None,
    interval: int = 1,
) → None
```

Register a state probe before calling `run()`.

| `name` | Target | Shape |
|--------|--------|-------|
| `"spikes"` | `population` | `(T, N)` or `(T, B, N)` |
| `"voltages"` | `population` | `(T, N)` or `(T, B, N)` |
| `"weights"` | `connection` | `(T//interval, out, in)` |

### `model.get_state()`

```python
model.get_state(
    name: str,
    population: Population = None,
    connection: Connection = None,
) → Tensor | ndarray
```

Retrieve recorded state after `run()`. Must call `record()` first.

### `model.metrics`

`dict` with performance metrics from the last `run()`:

| Key | Type | Description |
|-----|------|-------------|
| `"total_spikes"` | `int` | Total spike events |
| `"num_steps"` | `int` | Timesteps simulated |
| `"wall_ms"` | `float` | Wall time in milliseconds |

### `model.save()`

```python
model.save(path: str) → None
```

Save trained weights (GPU backend only). Writes a PyTorch checkpoint.

### `nuro.load()`

```python
nuro.load(path: str) → CompiledModel
```

Load a saved model. Requires the same graph definition.

---

## GPU-Specific: `model.snn`

The underlying `torch.nn.Module`. Available on GPU `CompiledModel`:

```python
# Access parameters
params = model.snn.parameters()
optimizer = torch.optim.Adam(params, lr=1e-3)

# Move to device
model.snn.to("cuda")
```

---

## NIR Interoperability

Import models from any NIR-compatible framework (SpikingJelly, Norse, snnTorch, Sinabs, Lava).

### `nuro.from_nir()`

```python
nuro.from_nir(nir_graph: nir.NIRGraph) -> IRGraph
```

Import a NIR graph into Nuro. Supports LIF, IF, CubaLIF neurons and Affine/Linear connections.

### `nuro.to_nir()`

```python
nuro.to_nir(graph: Graph | IRGraph) -> nir.NIRGraph
```

Export a Nuro graph to NIR format for use with other frameworks.

**Requires:** `pip install nuro[nir]`

---

## ANN-to-SNN Conversion

### `nuro.convert_ann()`

```python
nuro.convert_ann(
    model: nn.Module,
    input_shape: tuple,
    num_steps: int = 100,
) -> Graph
```

Convert a trained PyTorch ANN to a Nuro SNN. Supports:
- `nn.Linear` -> IF population + dense connection
- `nn.Conv2d` -> flattened dense connection
- `nn.ReLU` -> absorbed into IF threshold
- `nn.BatchNorm` -> folded into preceding weights

### `normalize_weights()`

```python
from nuro.conversion import normalize_weights

normalize_weights(
    graph: Graph,
    method: str = "robust",  # or "max"
    percentile: float = 99.0,
) -> Graph
```

Normalize connection weights for hardware deployment.

---

## Connectivity Patterns

| Pattern | Description | Params |
|---------|-------------|--------|
| `"dense"` | All-to-all (default) | — |
| `"random_sparse"` | Random with sparsity | `sparsity: float` |
| `"one_to_one"` | Diagonal (requires equal sizes) | — |
| `"conv1d"` | 1D convolutional | `kernel_size: int`, `stride: int` |
| `"distance_dependent"` | Gaussian distance-based | `sigma: float` |

---

## Synaptic Delays

```python
conn = nuro.Connection(
    source=pop1, target=pop2,
    delay=5e-3,  # 5ms delay
)
```

Spikes arrive at the target population after the specified delay. On hardware:
- **Loihi:** Native delay support
- **SpiNNaker 2:** Delay range [0, 7] timesteps

---

## Callbacks (MLOps)

```python
from nuro.callbacks import WandbCallback, TensorBoardCallback, PrintCallback

model.run(duration=0.1, callbacks=[
    WandbCallback(project="my-snn"),
    PrintCallback(log_interval=100),
])
```

---

## Datasets

```python
from nuro.datasets import NMNIST, DVSCifar10, DVSGesture

dataset = NMNIST(root="./data", train=True, num_steps=300)
spike_tensor, label = dataset[0]
```

---

## Supported Dynamics Summary

| `dynamics` | Description | Hardware support |
|------------|-------------|-----------------|
| `"lif"` | Leaky Integrate-and-Fire | GPU, Loihi, SpiNNaker, Akida |
| `"if"` | Integrate-and-Fire (no leak) | GPU, Loihi, SpiNNaker, Akida |
| `"izhikevich"` | Izhikevich (5 presets) | GPU, Loihi (sim), SpiNNaker (sim) |
| `"adex"` | Adaptive Exponential IF | GPU, Loihi (sim), SpiNNaker (sim) |

## Supported Backends

| Backend | Target | Install |
|---------|--------|---------|
| `"gpu"` | NVIDIA GPU / CPU | `pip install nuro[gpu]` |
| `"loihi"` | Intel Loihi 2 | `pip install nuro[loihi]` |
| `"spinnaker2"` | SpiNNaker 2 | `pip install nuro[spinnaker2]` |
| `"akida"` | BrainChip Akida | `pip install nuro[akida]` |
| `"cloud"` | Vantar Cloud | Built-in |
