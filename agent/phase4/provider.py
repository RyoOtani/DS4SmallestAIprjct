
import json
import os
import urllib.request
import urllib.error

class LLMProvider:
    """Abstract base class for LLM providers.
    
    Concrete implementations:
      - OpenAICompatibleProvider (HTTP API)
      - MockProvider (testing)
    
    Not meant to be instantiated directly.
    """
    def complete(self, messages, **kwargs):
        raise NotImplementedError("Use OpenAICompatibleProvider or MockProvider")

class OpenAICompatibleProvider(LLMProvider):
    """Works with OpenAI-compatible /v1/chat/completions endpoints."""
    def __init__(self, base_url=None, api_key=None, model=None, timeout=120):
        self.base_url = (base_url or os.getenv("TINYLLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("TINYLLM_API_KEY", "")
        self.model = model or os.getenv("TINYLLM_MODEL", "local-model")
        self.timeout = timeout

    def complete(self, messages, **kwargs):
        payload = {"model": self.model, "messages": messages}
        payload.update(kwargs)
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **({"Authorization": "Bearer "+self.api_key} if self.api_key else {})},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data=json.loads(r.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"LLM provider request failed: {e}") from e

class MockProvider(LLMProvider):
    def __init__(self, response="OK"):
        self.response=response
    def complete(self, messages, **kwargs):
        return self.response
