"""Train a spiking neural network to solve XOR using surrogate gradients.

This example demonstrates Nuro v0.4.0's gradient-based training:
- 2 input neurons, 10 hidden LIF neurons, 1 output neuron
- Spike count readout as prediction
- MSE loss with Adam optimizer
- Surrogate gradients (ATan) enable backpropagation through time

Usage::

    python examples/training/train_xor.py
"""

import torch

import nuro

# Network architecture
inp_pop = nuro.Population(size=2, dynamics="lif", params={"tau": 20e-3})
hidden = nuro.Population(size=10, dynamics="lif", params={"tau": 15e-3})
out_pop = nuro.Population(size=1, dynamics="lif", params={"tau": 10e-3})

c1 = nuro.Connection(source=inp_pop, target=hidden, pattern="dense")
c2 = nuro.Connection(source=hidden, target=out_pop, pattern="dense")

# XOR dataset: encode inputs as constant spike trains over 20 timesteps
# Target is the desired spike count of the output neuron
NUM_STEPS = 20
patterns = [
    (torch.zeros(NUM_STEPS, 2), 0.0),         # 0 XOR 0 = 0
    (torch.cat([torch.ones(NUM_STEPS, 1),
                torch.zeros(NUM_STEPS, 1)], dim=1), 5.0),  # 1 XOR 0 = 1
    (torch.cat([torch.zeros(NUM_STEPS, 1),
                torch.ones(NUM_STEPS, 1)], dim=1), 5.0),   # 0 XOR 1 = 1
    (torch.ones(NUM_STEPS, 2), 0.0),           # 1 XOR 1 = 0
]

NUM_EPOCHS = 100

print("Training XOR with surrogate gradients (ATan)")
print("=" * 50)

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0

    for data, target in patterns:
        # Build fresh graph each iteration (new compile to reset state)
        inp = nuro.Input(population=inp_pop, data=data)
        graph = nuro.Graph(
            [inp_pop, hidden, out_pop], [c1, c2], inputs=[inp]
        )
        model = nuro.compile(
            graph, target="gpu", requires_grad=True, surrogate="atan"
        )

        optimizer = torch.optim.Adam(model.snn.parameters(), lr=5e-3)
        optimizer.zero_grad()

        # Forward pass
        output = model.run(duration=NUM_STEPS * 1e-3, dt=1e-3)

        # Spike count readout
        pred = output[out_pop.id].sum()
        target_tensor = torch.tensor(target, device=pred.device)
        loss = (pred - target_tensor) ** 2

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
        print(f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f}")

print("=" * 50)
print("Training complete!")

# Test final predictions
print("\nFinal predictions:")
for data, target in patterns:
    inp = nuro.Input(population=inp_pop, data=data)
    graph = nuro.Graph(
        [inp_pop, hidden, out_pop], [c1, c2], inputs=[inp]
    )
    model = nuro.compile(
        graph, target="gpu", requires_grad=True, surrogate="atan"
    )
    output = model.run(duration=NUM_STEPS * 1e-3, dt=1e-3)
    pred = output[out_pop.id].sum().item()
    label = "HIGH" if target > 0 else "LOW"
    print(f"  Input: [{data[0, 0]:.0f}, {data[0, 1]:.0f}] | "
          f"Target: {label} ({target:.0f}) | "
          f"Spikes: {pred:.1f}")
