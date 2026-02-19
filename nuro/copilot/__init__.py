"""NeuroCopilot — AI coding assistant for the Nuro SDK.

Ask NeuroCopilot to generate Nuro Python code from natural language:

    import nuro
    code = nuro.copilot.ask("Build a LIF network for MNIST on Loihi 2")
    print(code)

Backends (in priority order):
    1. Ollama (local) — run ``ollama run vantar-ai/nuro-copilot`` first
    2. HuggingFace Inference API — set NURO_HF_TOKEN env var

Install Ollama backend::

    ollama pull vantar-ai/nuro-copilot
"""

from nuro.copilot.client import ask, CopilotClient

__all__ = ["ask", "CopilotClient"]
