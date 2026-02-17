# Contributing to Nuro

Thanks for your interest in Nuro. We're in early development and welcome contributions of all kinds.

## Getting Started

```bash
git clone https://github.com/Vantar-AI/nuro.git
cd nuro
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gpu,dev]"
pytest tests/ -v
```

## What We Need Help With

**High impact:**
- Testing with real SNN workloads and reporting what breaks
- New neuron models (Hodgkin-Huxley, ALIF, etc.)
- Performance benchmarks against SpikingJelly/Norse/BrainPy
- Loihi/SpiNNaker backend prototypes (if you have hardware access)

**Always welcome:**
- Bug reports with minimal reproduction steps
- Feature requests with use-case context
- Documentation improvements
- Test coverage for edge cases

## Development Workflow

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes and add tests
4. Run the test suite: `pytest tests/ -v`
5. Open a PR against `main`

## Code Style

- Python 3.10+ with type annotations
- We use `ruff` for formatting and linting: `ruff check . && ruff format .`
- Tests go in `tests/` and mirror the source structure
- Keep modules focused — one concept per file

## Project Structure

```
nuro/
  api/           # Developer-facing API (Population, Connection, Input, Graph)
  ir/            # Intermediate representation (DynamicsNode, SynapticEdge, IRGraph)
  backends/      # Compilation targets
    gpu/         # SpikingJelly GPU backend
  compiler/      # Compiler passes (stubs for now)
  runtime/       # Runtime execution (stubs for now)
tests/           # Test suite
examples/        # Runnable example scripts
```

## Adding a Neuron Model

1. Create or extend `nuro/backends/gpu/neurons.py` with a new `nn.Module`
2. Add the dynamics name to `SUPPORTED_DYNAMICS` in `nuro/api/population.py`
3. Wire it into `build_neuron_layer()` in `nuro/backends/gpu/dynamics.py`
4. Add tests in `tests/test_neuron_models.py`

## Adding a Backend

1. Create `nuro/backends/<name>/` with `backend.py` implementing `Backend` + `CompiledModel`
2. Register in `nuro/backends/__init__.py` `_REGISTRY`
3. Add tests in `tests/test_<name>_backend.py`

## Questions?

Open an issue or start a discussion. We're friendly.
