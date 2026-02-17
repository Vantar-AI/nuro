"""State Recording — inspecting voltages, spikes, and weights during simulation.

Demonstrates:
- Recording spike rasters
- Recording membrane voltages
- Recording weight snapshots at intervals
"""

import nuro

# Build a simple two-population network with STDP
input_pop = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3})
output_pop = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})
conn = nuro.Connection(
    source=input_pop,
    target=output_pop,
    pattern="dense",
    plasticity="stdp",
)

graph = nuro.Graph([input_pop, output_pop], [conn])
model = nuro.compile(graph, target="gpu")

# --- Register probes BEFORE running ---
model.record("spikes", population=input_pop)
model.record("spikes", population=output_pop)
model.record("voltages", population=output_pop)
model.record("weights", connection=conn, interval=100)  # every 100 steps

# Run for 1 second
model.run(duration=1.0, dt=1e-3)

# --- Retrieve recorded data ---
input_spikes = model.get_state("spikes", population=input_pop)
output_spikes = model.get_state("spikes", population=output_pop)
voltages = model.get_state("voltages", population=output_pop)
weights = model.get_state("weights", connection=conn)

print("=== State Recording Results ===")
print(f"Input spike raster:  shape {tuple(input_spikes.shape)}")
print(f"  (1000 timesteps x 50 neurons)")
print(f"Output spike raster: shape {tuple(output_spikes.shape)}")
print(f"  (1000 timesteps x 10 neurons)")
print(f"Output voltages:     shape {tuple(voltages.shape)}")
print(f"  (1000 timesteps x 10 neurons)")
print(f"Weight snapshots:    shape {tuple(weights.shape)}")
print(f"  (10 snapshots x 10 post x 50 pre)")

# Some statistics
print(f"\nTotal input spikes:  {int(input_spikes.sum().item())}")
print(f"Total output spikes: {int(output_spikes.sum().item())}")
print(f"Voltage range:       [{voltages.min().item():.3f}, {voltages.max().item():.3f}]")
print(f"Weight range (final): [{weights[-1].min().item():.4f}, {weights[-1].max().item():.4f}]")
