# Vantar Cloud

Remote compile and deploy to neuromorphic hardware — without owning a chip.

**Status: Beta (v0.7.0)**

---

## What Is It?

Vantar Cloud is a managed API for the Nuro SDK. Submit an IRGraph, get results back from real neuromorphic hardware:

```
Your laptop          Vantar Cloud API         Hardware
     |                      |                    |
     |-- POST /v1/compile -->|                    |
     |                      |-- compile + map --> |
     |<-- {job_id} ---------|                    |
     |                      |                    |
     |-- POST /v1/execute -->|                    |
     |                      |-- run on chip ----> |
     |<-- {metrics, spikes} |<-- results ---------|
```

**No INRC account needed. No board to maintain. Just an API key.**

---

## Quick Start

```python
import os
import nuro

# Set your API key
os.environ["VANTAR_API_KEY"] = "vt_your_key_here"

# Define and train your network as usual
inp  = nuro.Population(size=50, dynamics="lif", params={"tau": 20e-3})
out  = nuro.Population(size=10, dynamics="lif", params={"tau": 10e-3})
conn = nuro.Connection(source=inp, target=out, pattern="dense")
graph = nuro.Graph([inp, out], [conn])

# Train on GPU first
gpu_model = nuro.compile(graph, target="gpu", requires_grad=True)
# ... training loop ...
gpu_model.save("trained.pt")

# Deploy to Loihi via Vantar Cloud
cloud_model = nuro.compile(
    graph,
    target="cloud",
    hardware="loihi",           # or "spinnaker2"
    weights_from="trained.pt",
)

cloud_model.run(duration=1.0)
print(cloud_model.metrics)
```

---

## Supported Hardware

| `hardware=` | Chip | Notes |
|-------------|------|-------|
| `"loihi"` | Intel Loihi 2 | Default. LIF/IF native. |
| `"spinnaker2"` | SpiNNaker 2 | All LIF models supported. |

---

## Pricing Tiers (Beta)

| Tier | Price | Compile jobs | Hardware |
|------|-------|-------------|---------|
| **Free** | $0 | 10/month | Simulation only |
| **Developer** | $99/mo | Unlimited | Real hardware |
| **Team** | $499/mo | Unlimited + priority | Real hardware + SLA |

Beta access: [join the waitlist →](https://vantar.xyz)

---

## API Key

Get your key at [vantar.xyz](https://vantar.xyz) (waitlist).

Set via environment variable:
```bash
export VANTAR_API_KEY=vt_your_key_here
```

Or pass inline:
```python
model = nuro.compile(graph, target="cloud", api_key="vt_your_key_here", hardware="loihi")
```

---

## REST API

For direct API access (without the Nuro SDK):

**Base URL:** `https://api.vantar.xyz`

### Compile a graph

```
POST /v1/compile
Authorization: Bearer vt_your_key
Content-Type: application/json

{
  "ir_graph": { ... },   // IRGraph serialized to JSON
  "hardware": "loihi"    // "loihi" | "spinnaker2"
}
```

**Response:**
```json
{
  "job_id": "job_abc123",
  "status": "pending"
}
```

### Check status

```
GET /v1/jobs/{job_id}
Authorization: Bearer vt_your_key
```

**Response:**
```json
{
  "status": "compiled",   // "pending" | "compiling" | "compiled" | "error"
  "progress": 100,
  "error": null
}
```

### Execute

```
POST /v1/execute/{job_id}
Authorization: Bearer vt_your_key

{ "duration": 1.0 }
```

**Response:**
```json
{ "run_id": "run_xyz789" }
```

### Get results

```
GET /v1/results/{run_id}
Authorization: Bearer vt_your_key
```

**Response:**
```json
{
  "metrics": {
    "total_spikes": 4217,
    "num_steps": 1000,
    "wall_ms": 45.2
  },
  "spike_data": { ... }
}
```

---

## Architecture

```
nuro.compile(graph, target="cloud")
    ↓
CloudBackend.compile()
    ├── Serialize IRGraph → JSON
    └── POST /v1/compile → {job_id}

model.run(duration=1.0)
    ├── POST /v1/execute/{job_id}
    ├── Poll GET /v1/jobs/{job_id}
    └── GET /v1/results/{run_id} → metrics
```

On the server side:
```
API (FastAPI) → Worker → LoihiBroker / SpiNNaker2Broker → Hardware
```

Brokers install the full Nuro stack locally on the server and call the existing backends. No compilation logic duplication.

---

## Waitlist

Vantar Cloud is in beta. [Join the waitlist at vantar.xyz →](https://vantar.xyz)

Early access includes:
- Free tier (sim only, 10 compiles/month)
- Priority access to real hardware for waitlist members
- Direct feedback channel with the Nuro team
