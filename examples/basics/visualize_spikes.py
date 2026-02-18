"""Visualize spike activity — raster plot and voltage traces.

Demonstrates how to record and visualize:
- Spike raster plot (which neurons fired when)
- Membrane voltage traces (continuous dynamics)
- Per-neuron firing rates

Requirements:
    pip install nuro[gpu] matplotlib
"""

import torch
import nuro

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not installed. Run: pip install matplotlib")
    print("Continuing to demonstrate state recording API...\n")

# ── 1. Build a small network ─────────────────────────────────────────
inp_pop = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3, "v_thresh": -50.0})
out_pop = nuro.Population(size=20, dynamics="lif", params={"tau": 10e-3, "v_thresh": -50.0})

conn = nuro.Connection(source=inp_pop, target=out_pop, pattern="dense")

# Poisson input: 100 Hz average firing rate
inp = nuro.Input(population=inp_pop, mode="poisson", rate=100.0)

graph = nuro.Graph([inp_pop, out_pop], [conn], inputs=[inp])

# ── 2. Compile and register recorders ───────────────────────────────
model = nuro.compile(graph, target="gpu")

# Record spikes for the raster plot
model.record("spikes",   population=out_pop)

# Record membrane voltages for the trace plot
model.record("voltages", population=out_pop)

# ── 3. Run simulation ────────────────────────────────────────────────
DURATION = 0.5  # seconds
DT       = 1e-3  # 1 ms timestep

print(f"Running {DURATION * 1000:.0f} ms simulation...")
model.run(duration=DURATION, dt=DT)
print(f"Total spikes: {model.metrics['total_spikes']}")

# ── 4. Retrieve recorded state ────────────────────────────────────────
spikes   = model.get_state("spikes",   population=out_pop)   # (T, 20)
voltages = model.get_state("voltages", population=out_pop)   # (T, 20)

T, N = spikes.shape
t_ms = torch.arange(T) * (DT * 1e3)  # ms

# Per-neuron firing rate (Hz)
rates = spikes.sum(dim=0) / DURATION
print(f"Mean firing rate: {rates.mean():.1f} Hz  Max: {rates.max():.1f} Hz")

# ── 5. Plot ───────────────────────────────────────────────────────────
if not HAS_MATPLOTLIB:
    print(f"\nSpike array shape:   {spikes.shape}")
    print(f"Voltage array shape: {voltages.shape}")
    print("Install matplotlib to see the plots.")
    exit()

fig = plt.figure(figsize=(12, 8))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

# ── Raster plot ──────────────────────────────────────────────────────
ax_raster = fig.add_subplot(gs[0, :])
spike_np = spikes.numpy()
for neuron_idx in range(N):
    spike_times = t_ms[spike_np[:, neuron_idx] > 0.5].numpy()
    ax_raster.scatter(spike_times, [neuron_idx] * len(spike_times),
                      s=2, color="black", alpha=0.8)
ax_raster.set_xlabel("Time (ms)")
ax_raster.set_ylabel("Neuron index")
ax_raster.set_title("Spike Raster — Output Population (20 LIF neurons)")
ax_raster.set_xlim(0, DURATION * 1e3)
ax_raster.set_ylim(-0.5, N - 0.5)

# ── Voltage trace (first 3 neurons) ─────────────────────────────────
ax_volt = fig.add_subplot(gs[1, :])
colors = ["#e74c3c", "#2ecc71", "#3498db"]
volt_np = voltages.numpy()
for i, color in enumerate(colors):
    ax_volt.plot(t_ms.numpy(), volt_np[:, i], color=color,
                 linewidth=0.8, label=f"Neuron {i}", alpha=0.9)
ax_volt.set_xlabel("Time (ms)")
ax_volt.set_ylabel("Membrane potential (mV)")
ax_volt.set_title("Voltage Traces — First 3 Output Neurons")
ax_volt.set_xlim(0, DURATION * 1e3)
ax_volt.legend(fontsize=8, loc="upper right")

# ── Firing rate histogram ─────────────────────────────────────────────
ax_rates = fig.add_subplot(gs[2, 0])
ax_rates.bar(range(N), rates.numpy(), color="steelblue", alpha=0.8)
ax_rates.set_xlabel("Neuron index")
ax_rates.set_ylabel("Firing rate (Hz)")
ax_rates.set_title("Per-Neuron Firing Rates")

# ── Total spikes per timestep ─────────────────────────────────────────
ax_pop = fig.add_subplot(gs[2, 1])
pop_activity = spikes.sum(dim=1).numpy()
ax_pop.plot(t_ms.numpy(), pop_activity, color="darkorange", linewidth=0.8)
ax_pop.set_xlabel("Time (ms)")
ax_pop.set_ylabel("Active neurons")
ax_pop.set_title("Population Activity")
ax_pop.set_xlim(0, DURATION * 1e3)

plt.suptitle("Nuro — SNN State Visualization", fontsize=13, fontweight="bold", y=1.01)
plt.savefig("spike_visualization.png", dpi=150, bbox_inches="tight")
print("Saved plot: spike_visualization.png")
plt.show()
