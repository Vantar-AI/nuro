"""NeuroCopilot client — queries local Ollama or HuggingFace API."""

from __future__ import annotations

import os
import re
import json
from typing import Optional

SYSTEM_PROMPT = (
    "You are NeuroCopilot, an expert in the Nuro SDK for spiking neural networks (SNNs).\n"
    "When asked to write code, return only complete, runnable Python code that imports nuro "
    "and uses the Nuro SDK API. No placeholder comments. No explanations unless asked."
)

HF_MODEL = "VANTAR-AI/nuro-copilot-7b"
OLLAMA_MODEL = "vantar-ai/nuro-copilot"
OLLAMA_URL = "http://localhost:11434/api/generate"


def ask(
    prompt: str,
    *,
    backend: str = "auto",
    hf_token: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> str:
    """Ask NeuroCopilot to generate Nuro SDK code.

    Parameters
    ----------
    prompt:
        Natural language description of the task.
    backend:
        ``"ollama"`` (local), ``"hf"`` (HuggingFace API), or ``"auto"``
        (try Ollama first, fall back to HF).
    hf_token:
        HuggingFace API token. If not provided, reads ``NURO_HF_TOKEN``
        environment variable.
    temperature:
        Sampling temperature. Lower = more deterministic code.
    max_tokens:
        Maximum tokens to generate.

    Returns
    -------
    str
        Generated Python code (markdown fences stripped).

    Examples
    --------
    >>> import nuro
    >>> code = nuro.copilot.ask("Build a LIF network for MNIST on Loihi 2")
    >>> print(code)
    """
    client = CopilotClient(hf_token=hf_token)
    return client.ask(prompt, backend=backend, temperature=temperature, max_tokens=max_tokens)


class CopilotClient:
    """Stateful NeuroCopilot client (reuses HTTP sessions)."""

    def __init__(self, *, hf_token: Optional[str] = None):
        self._hf_token = hf_token or os.environ.get("NURO_HF_TOKEN")

    def ask(
        self,
        prompt: str,
        *,
        backend: str = "auto",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> str:
        if backend == "ollama":
            return self._ask_ollama(prompt, temperature=temperature, max_tokens=max_tokens)
        if backend == "hf":
            return self._ask_hf(prompt, temperature=temperature, max_tokens=max_tokens)
        # auto: try Ollama first
        try:
            return self._ask_ollama(prompt, temperature=temperature, max_tokens=max_tokens)
        except OllamaNotRunning:
            if self._hf_token:
                return self._ask_hf(prompt, temperature=temperature, max_tokens=max_tokens)
            raise RuntimeError(
                "NeuroCopilot: Ollama is not running and no NURO_HF_TOKEN is set.\n"
                "  Option 1 (local):  ollama pull vantar-ai/nuro-copilot\n"
                "                     ollama run vantar-ai/nuro-copilot\n"
                "  Option 2 (cloud):  export NURO_HF_TOKEN=<your-hf-token>"
            ) from None

    # ── Ollama ────────────────────────────────────────────────────────────────

    def _ask_ollama(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        try:
            import urllib.request
        except ImportError:
            raise RuntimeError("urllib not available")

        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode()

        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return _strip_fences(data.get("response", ""))
        except OSError as e:
            if "Connection refused" in str(e) or "refused" in str(e).lower():
                raise OllamaNotRunning() from e
            raise

    # ── HuggingFace Inference API ─────────────────────────────────────────────

    def _ask_hf(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if not self._hf_token:
            raise RuntimeError("NURO_HF_TOKEN not set")

        try:
            import urllib.request
        except ImportError:
            raise RuntimeError("urllib not available")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        payload = json.dumps({
            "inputs": _format_chatml(messages),
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False,
            },
        }).encode()

        url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._hf_token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            if isinstance(data, list):
                return _strip_fences(data[0].get("generated_text", ""))
            raise RuntimeError(f"Unexpected HF response: {data}")


# ── Helpers ───────────────────────────────────────────────────────────────────

class OllamaNotRunning(Exception):
    pass


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    text = text.strip()
    match = re.search(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _format_chatml(messages: list[dict]) -> str:
    """Format messages as ChatML string for HF text-generation pipeline."""
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
