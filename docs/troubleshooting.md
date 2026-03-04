# Troubleshooting

Common issues and solutions when working with Nuro.

---

## Installation Issues

### SpikingJelly not found

```
ModuleNotFoundError: No module named 'spikingjelly'
```

**Solution:** Install GPU extras: `pip install nuro[gpu]`

### Lava not found

```
ModuleNotFoundError: No module named 'lava'
```

**Solution:** Install Loihi extras: `pip install nuro[loihi]`

**Note:** Lava requires Python <= 3.10. If using 3.11+, create a separate venv.

### Akida not found

```
ModuleNotFoundError: No module named 'akida'
```

**Solution:** Install Akida extras: `pip install nuro[akida]`

BrainChip's SDK requires a separate license for hardware deployment.

---

## Compilation Errors

### "Unsupported dynamics" error

```
ValueError: Unsupported dynamics 'xyz'. Supported: ['adex', 'if', 'izhikevich', 'lif']
```

**Solution:** Use one of the supported dynamics strings: `"lif"`, `"if"`, `"izhikevich"`, `"adex"`.

### "Unknown backend" error

```
ValueError: Unknown backend 'xyz'. Available: ['akida', 'cloud', 'gpu', 'loihi', 'spinnaker2']
```

**Solution:** Use a valid target string in `nuro.compile()`.

### One-to-one pattern size mismatch

```
ValueError: one_to_one pattern requires equal population sizes
```

**Solution:** Ensure source and target populations have the same `size` when using `pattern="one_to_one"`.

---

## Training Issues

### No spikes during simulation

**Symptoms:** `model.metrics["total_spikes"] == 0`

**Causes & Solutions:**
1. **Input rate too low:** Increase `rate` in `nuro.Input(mode="poisson", rate=100.0)`
2. **Weights too small:** Nuro auto-scales weights, but if you set custom weights, ensure they're large enough to push membrane potential above threshold
3. **Timestep too large:** Try `dt=1e-4` instead of `dt=1e-3`
4. **tau too small:** A very small time constant causes rapid decay. Try `tau=20e-3`

### Exploding spike rates

**Symptoms:** Every neuron fires every timestep

**Solutions:**
1. **Reduce weight scale:** Lower the `weight_scale` in connection params
2. **Use inhibitory connections:** Add connections with negative weights
3. **Increase threshold:** Set `params={"v_thresh": 2.0}` on populations

### Gradient training produces NaN

**Solutions:**
1. **Lower learning rate:** Try `lr=1e-4`
2. **Clip gradients:** `torch.nn.utils.clip_grad_norm_(model.snn.parameters(), 1.0)`
3. **Use atan surrogate:** `nuro.compile(graph, requires_grad=True, surrogate="atan")`

---

## Hardware Deployment

### Weight transfer fails

```
KeyError: synapse key not found in checkpoint
```

**Solution:** Ensure the graph structure matches the checkpoint. The same `Population` and `Connection` objects must be used for both GPU training and hardware deployment.

### Loihi simulation crashes

**Solution:** Ensure `lava-nc` is installed and compatible:
```bash
pip install lava-nc==0.9.0
```

### SpiNNaker 2 delay out of range

SpiNNaker 2 supports delays in range [0, 7] timesteps. Values outside this range are clamped.

---

## Common Patterns

### Reset between epochs

```python
model.reset()
```

Always call `reset()` between training epochs to clear membrane potentials.

### Check model structure

```python
print(f"Populations: {graph.num_populations}")
print(f"Connections: {graph.num_connections}")
print(f"Cyclic: {graph.is_cyclic}")
```

### Debug spike activity

```python
model.record("spikes", population=pop)
model.run(duration=0.1)
spikes = model.get_state("spikes", population=pop)
print(f"Spike tensor shape: {spikes.shape}")
print(f"Total spikes: {spikes.sum()}")
```
