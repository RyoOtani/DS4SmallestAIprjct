"""Model provider abstraction for TinyLLM v2.

Supports local command adapters and OpenAI-compatible HTTP endpoints.
The C runtime remains the low-level inference core; this layer lets the
coding agent use a stronger external model when desired.
"""
from __future__ import annotations
import json, os, subprocess, urllib.request
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class ModelResponse:
    text: str
    raw: Any = None

class Provider:
    """Abstract base class for model providers.
    
    Concrete implementations:
      - CommandProvider (local CLI, e.g. ollama)
      - OpenAICompatibleProvider (HTTP API)
    Use provider_from_env() for automatic selection.
    """
    def generate(self, prompt: str, system: str = "") -> ModelResponse:
        raise NotImplementedError("Use CommandProvider or OpenAICompatibleProvider")

class OpenAICompatibleProvider(Provider):
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def generate(self, prompt: str, system: str = "") -> ModelResponse:
        payload = {"model": self.model, "messages": []}
        if system:
            payload["messages"].append({"role": "system", "content": system})
        payload["messages"].append({"role": "user", "content": prompt})
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode())
        return ModelResponse(data["choices"][0]["message"]["content"], data)

class CommandProvider(Provider):
    """Adapter for a local CLI model. Prompt is piped via stdin.
    
    Security: uses subprocess with shell=False and list arguments
    to prevent command injection. The command must be an executable path
    or a single command name (no shell metacharacters).
    """
    def __init__(self, command: str, timeout: int = 120):
        self.command = command
        self.timeout = timeout

    def generate(self, prompt: str, system: str = "") -> ModelResponse:
        full = (system + "\n\n" if system else "") + prompt
        import shlex
        cmd_parts = shlex.split(self.command)
        try:
            p = subprocess.run(
                cmd_parts,
                input=full, text=True,
                capture_output=True, timeout=self.timeout,
                shell=False,
            )
        except FileNotFoundError:
            raise RuntimeError(f"Command not found: {cmd_parts[0]}")
        if p.returncode:
            raise RuntimeError(p.stderr.strip() or f"provider exited {p.returncode}")
        return ModelResponse(p.stdout)

def provider_from_env() -> Provider:
    """Build provider from environment variables.

    TINYLLM_PROVIDER=openai_compatible (default) or command
    TINYLLM_BASE_URL=http://localhost:11434/v1
    TINYLLM_MODEL=...
    TINYLLM_API_KEY=...
    TINYLLM_COMMAND='ollama run ...'
    """
    kind = os.getenv("TINYLLM_PROVIDER", "openai_compatible")
    if kind == "command":
        return CommandProvider(os.environ["TINYLLM_COMMAND"])
    return OpenAICompatibleProvider(
        os.getenv("TINYLLM_BASE_URL", "http://localhost:11434/v1"),
        os.getenv("TINYLLM_MODEL", "qwen2.5-coder:7b"),
        os.getenv("TINYLLM_API_KEY", ""),
    )
