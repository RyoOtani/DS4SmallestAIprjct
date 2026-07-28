"""
providers.py — Abstract LLM Provider Architecture with Fallback Chains.

Architecture:
  BaseProvider (ABC)
  ├── TinyLLMProvider      — Self-hosted C inference server
  ├── OpenAICompatProvider  — OpenAI / DeepSeek / Groq / Ollama / etc.
  ├── AnthropicProvider     — Claude (Anthropic API)
  ├── GoogleProvider        — Gemini (Google AI API)
  ├── RuleBasedProvider     — Keyword/template matching (always works)
  └── FallbackProvider      — Chains multiple providers with failover

Typical fallback chain:
  TinyLLM (self) → DeepSeek → OpenAI → Groq → RuleBased (safe)
"""
from abc import ABC, abstractmethod
import json, os, re, time
from typing import List, Dict, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════
def _ok(text: str) -> Tuple[str, bool]: return (text, True)
def _fail(text: str) -> Tuple[str, bool]: return (text, False)

# ═══════════════════════════════════════════════════════════
# 1. Abstract Base Provider
# ═══════════════════════════════════════════════════════════
class BaseProvider(ABC):
    name: str = "base"
    label: str = "Base Provider"
    requires_api_key: bool = False

    @abstractmethod
    def chat(self, prompt: str, history: List[dict] = None,
             system_prompt: str = "", max_tokens: int = 1024,
             temperature: float = 0.7) -> Tuple[str, bool]:
        """Returns (response_text, success)."""
        ...

    def is_available(self) -> bool:
        return True

# ═══════════════════════════════════════════════════════════
# 2. TinyLLM Provider (self-hosted C inference)
# ═══════════════════════════════════════════════════════════
class TinyLLMProvider(BaseProvider):
    name = "tinyllm"
    label = "🦾 TinyLLM-nano (自前モデル)"
    requires_api_key = False

    def __init__(self, server_url: str = "http://localhost:8420", timeout: int = 120):
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            req = Request(f"{self.server_url}/health")
            with urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    def chat(self, prompt: str, history=None, system_prompt="",
             max_tokens=1024, temperature=0.7) -> Tuple[str, bool]:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        if history:
            ctx = "\n".join(
                f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                for m in history[-6:]
            )
            full_prompt = f"{ctx}\nUser: {prompt}\nAssistant:"
        try:
            body = json.dumps({"prompt": full_prompt, "max_tokens": max_tokens,
                               "temperature": temperature}).encode()
            req = Request(f"{self.server_url}/v1/completions", data=body,
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                text = data.get("text") or data.get("choices", [{}])[0].get("text", "")
                return _ok(text) if text.strip() else _fail("(TinyLLM empty)")
        except URLError as e:
            return _fail(f"TinyLLM unreachable: {e}")
        except Exception as e:
            return _fail(f"TinyLLM error: {e}")

# ═══════════════════════════════════════════════════════════
# 3. OpenAI-Compatible Provider
# ═══════════════════════════════════════════════════════════
class OpenAICompatProvider(BaseProvider):
    name = "openai_compat"
    label = "🤖 OpenAI互換API"
    requires_api_key = True

    def __init__(self, api_key="", base_url="", model="gpt-4o", label="", timeout=60):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        if label: self.label = label

    def is_available(self) -> bool:
        return bool(self.api_key and self.base_url)

    def chat(self, prompt: str, history=None, system_prompt="",
             max_tokens=1024, temperature=0.7) -> Tuple[str, bool]:
        if not self.api_key:
            return _fail("API key not set")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history: messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        try:
            body = json.dumps({"model": self.model, "messages": messages,
                               "max_tokens": max_tokens, "temperature": temperature}).encode()
            req = Request(f"{self.base_url}/chat/completions", data=body,
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {self.api_key}"})
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                text = data["choices"][0]["message"]["content"]
                return _ok(text) if text.strip() else _fail("(empty)")
        except HTTPError as e:
            err = e.read().decode(errors='replace')[:200]
            return _fail(f"HTTP {e.code}: {err}")
        except Exception as e:
            return _fail(f"API error: {e}")

# ═══════════════════════════════════════════════════════════
# 4. Anthropic Provider (Claude)
# ═══════════════════════════════════════════════════════════
class AnthropicProvider(BaseProvider):
    name = "anthropic"
    label = "🧠 Claude (Anthropic)"
    requires_api_key = True

    def __init__(self, api_key="", model="claude-sonnet-4-20250514", timeout=60):
        self.api_key = api_key; self.model = model; self.timeout = timeout

    def is_available(self) -> bool: return bool(self.api_key)

    def chat(self, prompt: str, history=None, system_prompt="",
             max_tokens=1024, temperature=0.7) -> Tuple[str, bool]:
        if not self.api_key: return _fail("Claude API key not set")
        messages = []
        if history:
            for m in history:
                messages.append({"role": m["role"], "content": m["content"]})
        messages.append({"role": "user", "content": prompt})
        try:
            body_data = {"model": self.model, "max_tokens": max_tokens, "messages": messages}
            if system_prompt: body_data["system"] = system_prompt
            body = json.dumps(body_data).encode()
            req = Request("https://api.anthropic.com/v1/messages", data=body,
                          headers={"Content-Type": "application/json",
                                   "x-api-key": self.api_key,
                                   "anthropic-version": "2023-06-01"})
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                text = data["content"][0]["text"]
                return _ok(text) if text.strip() else _fail("(Claude empty)")
        except Exception as e:
            return _fail(f"Claude error: {e}")

# ═══════════════════════════════════════════════════════════
# 5. Google Provider (Gemini)
# ═══════════════════════════════════════════════════════════
class GoogleProvider(BaseProvider):
    name = "google"
    label = "🌌 Gemini (Google)"
    requires_api_key = True

    def __init__(self, api_key="", model="gemini-2.5-flash", timeout=60):
        self.api_key = api_key; self.model = model; self.timeout = timeout

    def is_available(self) -> bool: return bool(self.api_key)

    def chat(self, prompt: str, history=None, system_prompt="",
             max_tokens=1024, temperature=0.7) -> Tuple[str, bool]:
        if not self.api_key: return _fail("Gemini API key not set")
        contents = []
        if history:
            for m in history:
                role = "user" if m["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        try:
            body_data = {"contents": contents,
                         "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}}
            if system_prompt:
                body_data["systemInstruction"] = {"parts": [{"text": system_prompt}]}
            body = json.dumps(body_data).encode()
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            req = Request(url, data=body, headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode())
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return _ok(text) if text.strip() else _fail("(Gemini empty)")
        except Exception as e:
            return _fail(f"Gemini error: {e}")

# ═══════════════════════════════════════════════════════════
# 6. Rule-Based Provider (always works)
# ═══════════════════════════════════════════════════════════
class RuleBasedProvider(BaseProvider):
    name = "rule_based"
    label = "🧩 ルールベース (安全処理)"
    requires_api_key = False

    def chat(self, prompt: str, history=None, system_prompt="",
             max_tokens=1024, temperature=0.7) -> Tuple[str, bool]:
        p = prompt.lower().strip()

        if any(w in p for w in ['こんにちは', 'hello', 'hi', 'hey', 'やあ']):
            return _ok("こんにちは！現在ルールベースモードです。⚙️設定からAPIキーを登録するとAIが応答します。")

        if any(w in p for w in ['何ができる', 'できること', '機能', 'help']):
            return _ok("🧩 ルールベースモード | 🌐 Web検索 | 🔀 マルチプロバイダー\n⚙️設定から DeepSeek/OpenAI/Claude/Gemini を接続可能")

        m = re.match(r'(\d+)\s*([\+\-\*/×÷])\s*(\d+)', p)
        if m:
            a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
            if op in ('+','＋'): r = a + b
            elif op in ('-','－','−'): r = a - b
            elif op in ('*','×','＊'): r = a * b
            elif op in ('/','÷','／'): r = a // b if b else "undefined"
            return _ok(f"{a} {op} {b} = **{r}**")

        if any(w in p for w in ['時間', 'いま', '今', 'time']):
            return _ok(f"現在時刻: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

        if any('\u3040' <= c <= '\u30ff' for c in prompt):
            return _ok(f"「{prompt[:100]}」についてのお問い合わせですね。\n\n⚠️ ルールベースモードです。⚙️設定からAPIキーを登録してください。")

        return _ok(f"Received: _{prompt[:200]}_\n\n⚠️ Rule-based mode. Add API key in ⚙️ Settings.")

# ═══════════════════════════════════════════════════════════
# 7. Fallback Provider (chains providers)
# ═══════════════════════════════════════════════════════════
class FallbackProvider(BaseProvider):
    """
    Chains providers. Tries each in order.
    All fail → returns safe minimum response.
    """
    name = "fallback"
    label = "🔀 自動フォールバック"
    requires_api_key = True

    def __init__(self, providers: List[BaseProvider] = None):
        self.providers = providers or []
        self._last: Optional[BaseProvider] = None

    def chat(self, prompt: str, history=None, system_prompt="",
             max_tokens=1024, temperature=0.7) -> Tuple[str, bool]:
        errors = []
        for i, p in enumerate(self.providers):
            if not p.is_available():
                errors.append(f"{p.label}: not available")
                continue
            try:
                text, ok = p.chat(prompt, history, system_prompt, max_tokens, temperature)
                if ok and text.strip():
                    self._last = p
                    if i > 0:
                        return _ok(f"*(via {p.label})*\n\n{text}")
                    return _ok(text)
                errors.append(f"{p.label}: {text}")
            except Exception as e:
                errors.append(f"{p.label}: {type(e).__name__}: {e}")

        self._last = None
        fallback = (
            "⚠️ **全AIモデルに接続できませんでした。**\n\n"
            + "\n".join(f"- {e}" for e in errors) +
            "\n\n💡 インターネット接続と⚙️設定のAPIキーを確認してください"
        )
        return _fail(fallback)

    @property
    def last_label(self) -> str:
        return self._last.label if self._last else "⚠️ 全失敗"

# ═══════════════════════════════════════════════════════════
# Presets
# ═══════════════════════════════════════════════════════════
PRESETS = {
    "tinyllm":      {"type": "tinyllm", "label": "🦾 TinyLLM-nano (自前1.5B)", "server_url": "http://localhost:8420"},
    "deepseek":     {"type": "openai_compat", "label": "🐋 DeepSeek Reasoner (R1)", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-reasoner"},
    "deepseek-chat":{"type": "openai_compat", "label": "🐋 DeepSeek Chat (V3)", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "openai":       {"type": "openai_compat", "label": "🤖 OpenAI (GPT-4o)", "base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
    "groq":         {"type": "openai_compat", "label": "⚡ Groq (Llama 3.1)", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.1-70b-versatile"},
    "openrouter":   {"type": "openai_compat", "label": "🔀 OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o"},
    "ollama":       {"type": "openai_compat", "label": "🦙 Ollama (ローカル)", "base_url": "http://localhost:11434/v1", "model": "llama3"},
    "claude":       {"type": "anthropic", "label": "🧠 Claude Sonnet 4", "model": "claude-sonnet-4-20250514"},
    "gemini":       {"type": "google", "label": "🌌 Gemini 2.5 Flash", "model": "gemini-2.5-flash"},
    "rule_based":   {"type": "rule_based", "label": "🧩 ルールベース (安全処理)"},
}

# ═══════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════
def create_provider(provider_id: str, api_key: str = "", base_url: str = "") -> BaseProvider:
    if provider_id.startswith("preset:"):
        provider_id = provider_id.split(":", 1)[1]
    preset = PRESETS.get(provider_id, {})
    ptype = preset.get("type", provider_id)

    if ptype == "tinyllm":
        return TinyLLMProvider(preset.get("server_url", "http://localhost:8420"))
    elif ptype == "openai_compat":
        return OpenAICompatProvider(
            api_key=api_key,
            base_url=base_url or preset.get("base_url", "https://api.openai.com/v1"),
            model=preset.get("model", "gpt-4o"),
            label=preset.get("label", ""),
        )
    elif ptype == "anthropic":
        return AnthropicProvider(api_key=api_key, model=preset.get("model", "claude-sonnet-4-20250514"))
    elif ptype == "google":
        return GoogleProvider(api_key=api_key, model=preset.get("model", "gemini-2.5-flash"))
    elif ptype == "rule_based":
        return RuleBasedProvider()
    else:
        return RuleBasedProvider()

def create_fallback_chain(api_key: str = "",
                          tinyllm_url: str = "http://localhost:8420") -> FallbackProvider:
    return FallbackProvider([
        TinyLLMProvider(tinyllm_url),
        OpenAICompatProvider(api_key, "https://api.deepseek.com/v1", "deepseek-reasoner", "🐋 DeepSeek R1"),
        OpenAICompatProvider(api_key, "https://api.openai.com/v1", "gpt-4o", "🤖 OpenAI GPT-4o"),
        OpenAICompatProvider(api_key, "https://api.groq.com/openai/v1", "llama-3.1-70b-versatile", "⚡ Groq"),
        RuleBasedProvider(),
    ])

def list_providers() -> List[dict]:
    providers = [
        {"id": "fallback", "label": "🔀 自動フォールバック (推奨)", "requires_api_key": True},
        {"id": "tinyllm", "label": "🦾 TinyLLM-nano (自前1.5B)", "requires_api_key": False},
    ]
    for pid in ["deepseek", "deepseek-chat", "openai", "groq", "openrouter", "ollama", "claude", "gemini"]:
        p = PRESETS.get(pid, {})
        providers.append({"id": pid, "label": p.get("label", pid), "requires_api_key": p.get("type") != "tinyllm"})
    providers.append({"id": "rule_based", "label": "🧩 ルールベース (安全処理)", "requires_api_key": False})
    return providers
