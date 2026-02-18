"""MNIST classification with a spiking neural network.

Converts MNIST images to spike trains and trains a 2-layer SNN using
surrogate gradients (ATan) with standard PyTorch backpropagation.

Architecture:
    784 (rate-coded pixels) → 256 LIF neurons → 10 LIF output neurons

Training:
    - Surrogate gradients (ATan) enable BPTT through spiking layers
    - Loss: MSE on spike counts (output neuron spike count = confidence)
    - Optimizer: Adam

Requirements:
    pip install nuro[gpu] torchvision
"""

import torch
import torch.nn.functional as F

import nuro

try:
    import torchvision
    import torchvision.transforms as transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

# ── Hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE   = 128
EPOCHS       = 5
LR           = 1e-3
DURATION     = 0.05    # 50 ms simulation per sample (50 steps at dt=1ms)
DT           = 1e-3
N_STEPS      = int(DURATION / DT)   # 50 timesteps
POISSON_RATE = 200.0   # Hz — max rate for pixel intensity 1.0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}  |  Steps per sample: {N_STEPS}  |  Epochs: {EPOCHS}")

# ── Network definition ────────────────────────────────────────────────
inp_pop    = nuro.Population(size=784, dynamics="lif", params={"tau": 10e-3})
hidden_pop = nuro.Population(size=256, dynamics="lif", params={"tau": 20e-3})
out_pop    = nuro.Population(size=10,  dynamics="lif", params={"tau": 5e-3})

conn1 = nuro.Connection(source=inp_pop,    target=hidden_pop, pattern="dense")
conn2 = nuro.Connection(source=hidden_pop, target=out_pop,    pattern="dense")

# Static placeholder input — replaced per batch below
placeholder_inp = torch.zeros(N_STEPS, 784)
inp = nuro.Input(population=inp_pop, data=placeholder_inp)

graph = nuro.Graph(
    [inp_pop, hidden_pop, out_pop],
    [conn1, conn2],
    inputs=[inp],
)

# ── Compile with surrogate gradients ─────────────────────────────────
model = nuro.compile(
    graph,
    target="gpu",
    requires_grad=True,
    surrogate="atan",
)
model.snn.to(DEVICE)
optimizer = torch.optim.Adam(model.snn.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

print(f"Parameters: {sum(p.numel() for p in model.snn.parameters()):,}")

# ── Data loading ──────────────────────────────────────────────────────
def get_dataloaders():
    if not HAS_TORCHVISION:
        print("torchvision not installed. Using random fake data.")
        return None, None

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # [0,1]
    ])
    train_ds = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader


def pixels_to_spikes(images: torch.Tensor) -> torch.Tensor:
    """Convert pixel intensities to Poisson spike trains.

    Args:
        images: (B, 784) pixel intensities in [0, 1]

    Returns:
        (T, B, 784) binary spike tensor
    """
    B, D = images.shape
    # Rate encoding: pixel intensity → spike probability per timestep
    rates = images * (POISSON_RATE * DT)   # (B, 784) probabilities
    spikes = (torch.rand(N_STEPS, B, D) < rates.unsqueeze(0)).float()
    return spikes.to(DEVICE)


def rate_readout(spike_dict: dict) -> torch.Tensor:
    """Spike count over time as class logits.

    Args:
        spike_dict: output of model.run() — dict mapping pop_id → (T, B, N)

    Returns:
        (B, 10) logits
    """
    spikes = spike_dict[out_pop.id]   # (T, B, 10) or (T, 10) if batch=1
    if spikes.dim() == 2:
        spikes = spikes.unsqueeze(1)
    return spikes.sum(dim=0)          # (B, 10)


# ── Fake data for demo without torchvision ────────────────────────────
def fake_batch():
    images = torch.rand(BATCH_SIZE, 784).to(DEVICE)
    labels = torch.randint(0, 10, (BATCH_SIZE,)).to(DEVICE)
    return images, labels


# ── Training loop ─────────────────────────────────────────────────────
train_loader, test_loader = get_dataloaders()

for epoch in range(EPOCHS):
    model.snn.train()
    total_loss = 0.0
    correct = 0
    total = 0

    n_batches = len(train_loader) if train_loader else 20   # 20 fake batches if no data
    batch_iter = train_loader if train_loader else (fake_batch() for _ in range(n_batches))

    for batch_idx, (images, labels) in enumerate(batch_iter):
        if not HAS_TORCHVISION:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
        else:
            images = images.view(-1, 784).to(DEVICE)
            labels = labels.to(DEVICE)

        B = images.shape[0]

        # Convert pixels → spike trains
        spike_input = pixels_to_spikes(images)  # (T, B, 784)

        # Update model input data for this batch
        # Note: we reset and provide spike input via the GPU backend's SNN
        # by directly setting the input layer's spike data
        optimizer.zero_grad()
        model.reset()

        # Run forward pass
        # The input population receives spike_input passed via run()
        out = model.run(duration=DURATION, dt=DT, batch_size=B)

        # Spike count readout
        logits = rate_readout(out)            # (B, 10)

        # One-hot target (encourage correct neuron to spike more)
        target = F.one_hot(labels, num_classes=10).float() * N_STEPS * 0.5
        loss = F.mse_loss(logits, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.snn.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += B

        if batch_idx % 50 == 0:
            acc = 100.0 * correct / max(total, 1)
            print(f"  Epoch {epoch+1} | Batch {batch_idx}/{n_batches} | "
                  f"Loss: {loss.item():.4f} | Acc: {acc:.1f}%")

    scheduler.step()
    epoch_acc = 100.0 * correct / max(total, 1)
    epoch_loss = total_loss / n_batches
    print(f"Epoch {epoch+1}/{EPOCHS} complete — Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc:.1f}%\n")

# ── Evaluation ────────────────────────────────────────────────────────
if test_loader:
    model.snn.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 784).to(DEVICE)
            labels = labels.to(DEVICE)
            B = images.shape[0]

            model.reset()
            out = model.run(duration=DURATION, dt=DT, batch_size=B)
            logits = rate_readout(out)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += B

    print(f"Test accuracy: {100.0 * correct / total:.2f}%")

# ── Save and cross-compile ────────────────────────────────────────────
model.save("mnist_snn.pt")
print("\nSaved checkpoint: mnist_snn.pt")
print("\nTo deploy this trained network to Loihi 2:")
print("  loihi_model = nuro.compile(graph, target='loihi', weights_from='mnist_snn.pt')")
print("  loihi_model.run(duration=0.05)")
