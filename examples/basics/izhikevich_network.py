"""Izhikevich Network — biologically realistic spiking neurons.

Demonstrates:
- Izhikevich neuron model with different firing patterns
- AdEx (Adaptive Exponential IF) neurons
- Mixed networks combining LIF and biologically realistic models
"""

import nuro

# --- 1. Regular Spiking Izhikevich network ---
print("=== Regular Spiking Izhikevich ===")

input_pop = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
izh_pop = nuro.Population(size=20, dynamics="izhikevich")
conn = nuro.Connection(source=input_pop, target=izh_pop, pattern="dense")

graph = nuro.Graph([input_pop, izh_pop], [conn])
model = nuro.compile(graph, target="gpu")
model.record("spikes", population=izh_pop)
model.run(duration=0.5, dt=1e-3)

spikes = model.get_state("spikes", population=izh_pop)
print(f"  Izh spikes: {int(spikes.sum().item())} over 500 steps")
print(f"  Avg firing rate: {spikes.sum().item() / (20 * 0.5):.1f} Hz")

# --- 2. Fast Spiking Izhikevich ---
print("\n=== Fast Spiking Izhikevich ===")

input_pop2 = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
fs_pop = nuro.Population(
    size=20,
    dynamics="izhikevich",
    params={"preset": "fast_spiking"},
)
conn2 = nuro.Connection(source=input_pop2, target=fs_pop, pattern="dense")

graph2 = nuro.Graph([input_pop2, fs_pop], [conn2])
model2 = nuro.compile(graph2, target="gpu")
model2.record("spikes", population=fs_pop)
model2.run(duration=0.5, dt=1e-3)

spikes2 = model2.get_state("spikes", population=fs_pop)
print(f"  FS spikes: {int(spikes2.sum().item())} over 500 steps")
print(f"  Avg firing rate: {spikes2.sum().item() / (20 * 0.5):.1f} Hz")

# --- 3. AdEx neurons ---
print("\n=== Adaptive Exponential IF (AdEx) ===")

input_pop3 = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
adex_pop = nuro.Population(size=20, dynamics="adex")
conn3 = nuro.Connection(source=input_pop3, target=adex_pop, pattern="dense")

graph3 = nuro.Graph([input_pop3, adex_pop], [conn3])
model3 = nuro.compile(graph3, target="gpu")
model3.record("spikes", population=adex_pop)
model3.run(duration=0.5, dt=1e-3)

spikes3 = model3.get_state("spikes", population=adex_pop)
print(f"  AdEx spikes: {int(spikes3.sum().item())} over 500 steps")
print(f"  Avg firing rate: {spikes3.sum().item() / (20 * 0.5):.1f} Hz")

# --- 4. Mixed network: LIF → Izhikevich → AdEx ---
print("\n=== Mixed Network (LIF → Izh → AdEx) ===")

pop_lif = nuro.Population(size=100, dynamics="lif", params={"tau": 20e-3})
pop_izh = nuro.Population(size=50, dynamics="izhikevich", params={"preset": "chattering"})
pop_adex = nuro.Population(size=20, dynamics="adex")

c1 = nuro.Connection(source=pop_lif, target=pop_izh, pattern="dense")
c2 = nuro.Connection(source=pop_izh, target=pop_adex, pattern="dense")

graph4 = nuro.Graph([pop_lif, pop_izh, pop_adex], [c1, c2])
model4 = nuro.compile(graph4, target="gpu")
model4.run(duration=0.5, dt=1e-3)

print(f"  Total spikes: {model4.metrics['total_spikes']}")
for name, pop in [("LIF", pop_lif), ("Izh", pop_izh), ("AdEx", pop_adex)]:
    count = model4.metrics["spike_counts"][pop.id]
    print(f"  {name}: {count} spikes ({count / (pop.size * 0.5):.1f} Hz avg)")
