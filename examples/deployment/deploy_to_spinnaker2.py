"""Deploy to SpiNNaker 2 — train on GPU, deploy to neuromorphic hardware.

Mirrors the deploy_to_loihi.py workflow but targeting SpiNNaker 2:
1. Define a network with Nuro's API
2. Train with surrogate gradients on GPU
3. Save the trained weights
4. Recompile to SpiNNaker 2 backend with weight transfer
5. Run inference on SpiNNaker 2 (Brian2 simulator or real hardware)

Requirements:
    pip install nuro[gpu,spinnaker2]

SpiNNaker 2 hardware access:
    https://spinncloud.com — SpiNNcloud public access
"""

import numpy as np

import nuro

# ── 1. Define the network ────────────────────────────────────────────
sensory = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3})
hidden  = nuro.Population(size=30, dynamics="lif", params={"tau": 15e-3})
motor   = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})

conn1 = nuro.Connection(source=sensory, target=hidden, pattern="dense")
conn2 = nuro.Connection(source=hidden,  target=motor,  pattern="dense")

data = (np.random.rand(100, 50) < 0.1).astype(np.float32)
inp = nuro.Input(population=sensory, data=data)

graph = nuro.Graph(
    [sensory, hidden, motor],
    [conn1, conn2],
    inputs=[inp],
)

# ── 2. Train on GPU with surrogate gradients ─────────────────────────
print("Training on GPU...")

try:
    import torch

    gpu_model = nuro.compile(graph, target="gpu", requires_grad=True)
    optimizer = torch.optim.Adam(gpu_model.snn.parameters(), lr=1e-3)

    for epoch in range(5):
        optimizer.zero_grad()
        output = gpu_model.run(duration=0.1, dt=1e-3)
        # Encourage motor population to spike
        loss = -output[motor.id].sum()
        loss.backward()
        optimizer.step()
        gpu_model.reset()
        print(f"  Epoch {epoch + 1}: loss = {loss.item():.4f}")

    gpu_model.save("trained_network_sp2.pt")
    print("Saved GPU checkpoint: trained_network_sp2.pt")

except ImportError:
    print("PyTorch/SpikingJelly not available — skipping GPU training.")
    print("Compiling directly to SpiNNaker 2 (no pre-trained weights)...")
    sp2_model = nuro.compile(graph, target="spinnaker2")
    sp2_model.run(duration=0.1)
    print(f"SpiNNaker 2 sim metrics: {sp2_model.metrics}")
    exit()

# ── 3. Deploy to SpiNNaker 2 ─────────────────────────────────────────
print("\nDeploying to SpiNNaker 2 (Brian2 simulation mode)...")

sp2_model = nuro.compile(
    graph,
    target="spinnaker2",
    weights_from="trained_network_sp2.pt",
)

# Record spikes on motor population
sp2_model.record("spikes", population=motor)

# Run inference — uses Brian2 by default (no hardware needed)
sp2_model.run(duration=0.1)

print(f"Total spikes:      {sp2_model.metrics['total_spikes']}")
print(f"Simulation steps:  {sp2_model.metrics['num_steps']}")

spikes = sp2_model.get_state("spikes", population=motor)
print(f"Motor spike shape: {spikes.shape}")   # (100, 10)
print(f"Motor spike count: {int(np.sum(spikes))}")

# ── 4. Switch to real SpiNNaker 2 hardware ───────────────────────────
# Requires SpiNNcloud access or a local SpiNNaker 2 board.
# Uncomment the lines below:
#
# sp2_model.set_hardware(True)   # switches from Brian2 sim to real chip
# sp2_model.run(duration=1.0)    # runs on neuromorphic silicon
#
# Or connect to SpiNNcloud:
# import spinnaker2
# board = spinnaker2.SpiNNaker2Board(host="your-board-ip")
# sp2_model.set_hardware(True, board=board)

print("\nDone! Train on GPU → Deploy to SpiNNaker 2 in 3 lines of code.")
