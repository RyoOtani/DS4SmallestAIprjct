"""
providers.py — Multi-backend LLM providers for TinyLLM Chat.

Supported backends:
  - openai_compatible : OpenAI / DeepSeek / Groq / local LLM with OpenAI API
  - tinyllm_c         : TinyLLM C inference server (src/http.c, port 8420)
  - template          : Template-based responses (no model, offline demo)
"""

import json, os, re
from pathlib import Path
from typing import List, Dict, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


# ═══════════════════════════════════════════════════════════
# Base Provider
# ═══════════════════════════════════════════════════════════

class BaseProvider:
    """Abstract base for all LLM backends."""
    name: str = "base"
    label: str = "Base"

    def chat(self, prompt: str, history: List[dict] = None,
             system_prompt: str = "", max_tokens: int = 1024,
             temperature: float = 0.7) -> str:
        raise NotImplementedError


# ═══════════════════════════════════════════════════════════
# Template Provider (no model — current behavior)
# ═══════════════════════════════════════════════════════════

class TemplateProvider(BaseProvider):
    """Keyword-matching template responses. No LLM, works offline."""
    name = "template"
    label = "🧩 テンプレート (オフライン)"

    def chat(self, prompt: str, history: List[dict] = None,
             system_prompt: str = "", max_tokens: int = 1024,
             temperature: float = 0.7) -> str:
        p = prompt.lower()

        if any(w in p for w in ['こんにちは', 'hello', 'hi', 'hey']):
            return "こんにちは！TinyLLM チャットへようこそ。右下の⚙️設定からモデルを切り替えられます。何かお手伝いしましょうか？"

        if any(w in p for w in ['何ができる', 'できること', '機能', 'help']):
            return """🌐 **Web検索**: DuckDuckGo でリアルタイム検索
🧠 **エージェントモード**: 複数ステップの推論
🤖 **モデル切替**: OpenAI / DeepSeek / TinyLLM を選択可能
🔑 **APIキー**: 設定画面から自分のAPIキーを登録

右下の ⚙️ アイコンから設定を開けます。"""

        if any(w in p for w in ['コード', 'code', 'python', '関数']):
            return '```python\ndef hello(name: str) -> str:\n    return f"こんにちは、{name}さん！"\n```\n\nより高度なコード生成には、OpenAI か TinyLLM モデルを選択してください。'

        if any('\u3040' <= c <= '\u30ff' for c in prompt):
            return f"「{prompt[:100]}」についてのお問い合わせですね。\n\n⚙️設定から **OpenAI** や **TinyLLM** モデルを選択すると、AIが直接回答を生成します。"

        return f"I understand: _{prompt[:200]}_\n\n💡 Open ⚙️ Settings to connect an AI model for intelligent responses."


# ═══════════════════════════════════════════════════════════
# OpenAI-Compatible Provider
# ═══════════════════════════════════════════════════════════

class OpenAICompatibleProvider(BaseProvider):
    """
    OpenAI / DeepSeek / Groq / local (Ollama, vLLM) — anything with
    a /v1/chat/completions endpoint.
    """
    name = "openai_compatible"
    label = "🤖 OpenAI互換API"

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = "", label: str = ""):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        if label:
            self.label = label

    def chat(self, prompt: str, history: List[dict] = None,
             system_prompt: str = "", max_tokens: int = 1024,
             temperature: float = 0.7) -> str:
        if not self.api_key:
            return "⚠️ APIキーが設定されていません。⚙️設定からAPIキーを入力してください。"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        try:
            req = Request(
                f"{self.base_url}/chat/completions",
                data=body.encode(),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
            )
            with urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
                return data["choices"][0]["message"]["content"]
        except URLError as e:
            return f"⚠️ API接続エラー: {e}\n\nbase_url=`{self.base_url}` が正しいか確認してください。"
        except Exception as e:
            return f"⚠️ APIエラー: {e}"


# ═══════════════════════════════════════════════════════════
# TinyLLM C Inference Server Provider
# ═══════════════════════════════════════════════════════════

class TinyLLMCProvider(BaseProvider):
    """Connect to TinyLLM C inference server (src/http.c)."""
    name = "tinyllm_c"
    label = "🦾 TinyLLM (C推論サーバー)"

    def __init__(self, server_url: str = "http://localhost:8420"):
        self.server_url = server_url.rstrip('/')

    def chat(self, prompt: str, history: List[dict] = None,
             system_prompt: str = "", max_tokens: int = 1024,
             temperature: float = 0.7) -> str:
        # Format with system prompt for the raw completion endpoint
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"

        # Build conversation context from history
        if history:
            ctx_parts = []
            for msg in history[-6:]:  # last 6 messages
                role = "User" if msg["role"] == "user" else "Assistant"
                ctx_parts.append(f"{role}: {msg['content']}")
            full_prompt = "\n".join(ctx_parts) + f"\nUser: {prompt}\nAssistant:"

        body = json.dumps({
            "prompt": full_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        try:
            req = Request(
                f"{self.server_url}/v1/completions",
                data=body.encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                # C server returns {"text": "..."} or {"choices": [{"text": "..."}]}
                if "text" in data:
                    return data["text"]
                if "choices" in data:
                    return data["choices"][0].get("text", "")
                return json.dumps(data, ensure_ascii=False)
        except URLError as e:
            return f"⚠️ TinyLLMサーバーに接続できません: {e}\n\n`make run` でC推論サーバーを起動してください。(port {self.server_url.split(':')[-1]})"
        except Exception as e:
            return f"⚠️ TinyLLMエラー: {e}"


# ═══════════════════════════════════════════════════════════
# Provider Registry & Factory
# ═══════════════════════════════════════════════════════════

BUILTIN_PROVIDERS: Dict[str, type] = {
    "template": TemplateProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "tinyllm_c": TinyLLMCProvider,
}

# Preset configurations for popular APIs
PRESETS = {
    "openai": {
        "type": "openai_compatible",
        "label": "🤖 OpenAI (GPT-4o)",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    "deepseek": {
        "type": "openai_compatible",
        "label": "🐋 DeepSeek Reasoner (R1)",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-reasoner",
    },
    "deepseek-chat": {
        "type": "openai_compatible",
        "label": "🐋 DeepSeek Chat (V3)",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
    },
    "groq": {
        "type": "openai_compatible",
        "label": "⚡ Groq (Llama 3)",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-70b-versatile",
    },
    "openrouter": {
        "type": "openai_compatible",
        "label": "🔀 OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o",
    },
    "tinyllm": {
        "type": "tinyllm_c",
        "label": "🦾 TinyLLM-nano (1.5B)",
        "server_url": "http://localhost:8420",
    },
    "ollama": {
        "type": "openai_compatible",
        "label": "🦙 Ollama (ローカル)",
        "base_url": "http://localhost:11434/v1",
        "model": "llama3",
    },
}


def create_provider(provider_type: str, **kwargs) -> BaseProvider:
    """Factory: create a provider instance from type and config."""
    if provider_type in BUILTIN_PROVIDERS:
        cls = BUILTIN_PROVIDERS[provider_type]
        if provider_type == "openai_compatible":
            return cls(
                api_key=kwargs.get("api_key", ""),
                base_url=kwargs.get("base_url", "https://api.openai.com/v1"),
                model=kwargs.get("model", "gpt-4o"),
                label=kwargs.get("label", ""),
            )
        elif provider_type == "tinyllm_c":
            return cls(server_url=kwargs.get("server_url", "http://localhost:8420"))
        else:
            return cls()
    # Fallback to template
    return TemplateProvider()


def create_from_preset(preset_name: str, api_key: str = "") -> BaseProvider:
    """Create a provider from a named preset + user's API key."""
    preset = PRESETS.get(preset_name, {})
    ptype = preset.get("type", "template")
    kwargs = {k: v for k, v in preset.items() if k not in ("type", "label")}
    kwargs["api_key"] = api_key
    kwargs["label"] = preset.get("label", "")
    return create_provider(ptype, **kwargs)


def list_providers() -> List[dict]:
    """List all available providers for the UI."""
    providers = []
    # Built-in types
    for name, cls in BUILTIN_PROVIDERS.items():
        providers.append({"id": name, "label": cls.label, "requires_api_key": name == "openai_compatible"})
    # Presets
    for name, preset in PRESETS.items():
        providers.append({
            "id": f"preset:{name}",
            "label": preset.get("label", name),
            "requires_api_key": preset.get("type") == "openai_compatible",
            "preset": name,
        })
    return providers
