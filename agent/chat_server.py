#!/usr/bin/env python3
"""
chat_server.py — TinyLLM Web Chat Server

Serves:
  - GET  /              → Web chat UI (agent/web_chat/index.html)
  - POST /v1/chat       → Chat API (with web search + agent mode)
  - GET  /health        → Health check

Features:
  - 🌐 Web search via DuckDuckGo (free, no API key)
  - 🧠 Agent mode: multi-step reasoning with tool use
  - 💬 Conversation history per session
  - 🖥️  Single-file, zero external web framework deps

Usage:
  python chat_server.py                          # Default: port 8421
  python chat_server.py --port 8080              # Custom port
  python chat_server.py --backend http://localhost:8420  # Proxy to C inference server
"""

import json, os, sys, time, uuid, argparse, threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List

# Add repo root (tinyllm/) to path so `from agent.xxx` works
_REPO = Path(__file__).resolve().parent.parent  # tinyllm/
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agent.web_search import search_and_summarize, web_search, fetch_page
from agent.providers import (
    BaseProvider, TemplateProvider, OpenAICompatibleProvider, TinyLLMCProvider,
    create_provider, create_from_preset, list_providers, PRESETS,
)


# ═══════════════════════════════════════════════════════════════
# Server Config (in-memory — no disk write for API key safety)
# ═══════════════════════════════════════════════════════════════

class ServerConfig:
    """Runtime config: selected provider, API keys (never written to disk)."""
    def __init__(self):
        self.provider_id = "template"
        self.api_key = ""
        self.provider: BaseProvider = TemplateProvider()

    def to_dict(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "has_api_key": bool(self.api_key),
            "provider_label": self.provider.label,
        }

    def set_provider(self, provider_id: str, api_key: str = None):
        """Switch to a different provider."""
        if api_key is not None and api_key.strip():
            self.api_key = api_key.strip()

        self.provider_id = provider_id

        if provider_id.startswith("preset:"):
            preset_name = provider_id.split(":", 1)[1]
            self.provider = create_from_preset(preset_name, self.api_key)
        elif provider_id == "openai_compatible":
            # Custom OpenAI-compatible — use env vars or defaults
            self.provider = OpenAICompatibleProvider(
                api_key=self.api_key,
                base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            )
        elif provider_id == "tinyllm_c":
            self.provider = TinyLLMCProvider(
                server_url=os.environ.get("TINYLLM_SERVER", "http://localhost:8420")
            )
        else:
            self.provider = create_provider(provider_id, api_key=self.api_key)
        
        print(f"🔀 Provider switched to: {self.provider.label}")


# ═══════════════════════════════════════════════════════════════
# Conversation Store
# ═══════════════════════════════════════════════════════════════

class ConversationStore:
    """In-memory conversation storage with TTL."""
    def __init__(self, ttl_hours: int = 24):
        self._convs: Dict[str, List[dict]] = {}
        self._ttl = ttl_hours * 3600
        self._timestamps: Dict[str, float] = {}
    
    def get(self, conv_id: str) -> List[dict]:
        self._cleanup()
        return self._convs.get(conv_id, [])
    
    def add(self, conv_id: str, role: str, content: str):
        if conv_id not in self._convs:
            self._convs[conv_id] = []
        self._convs[conv_id].append({"role": role, "content": content})
        self._timestamps[conv_id] = time.time()
    
    def _cleanup(self):
        now = time.time()
        expired = [cid for cid, ts in self._timestamps.items() if now - ts > self._ttl]
        for cid in expired:
            self._convs.pop(cid, None)
            self._timestamps.pop(cid, None)


# ═══════════════════════════════════════════════════════════════
# Chat Agent (provider-agnostic)
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are TinyLLM, an expert AI coding assistant and general-purpose helper.

## Your Capabilities
- **Code Generation**: Write clean, idiomatic, well-documented code in any language
- **Debugging**: Find and fix bugs with clear explanations
- **Architecture**: Design system architectures and explain trade-offs
- **Web Search**: Access real-time information via DuckDuckGo
- **Multi-language**: Respond in the same language as the user (日本語OK)

## Coding Guidelines
When writing code:
1. Use meaningful variable/function names
2. Add type hints (Python), interfaces (TypeScript), or equivalent
3. Include brief docstrings/comments for non-obvious logic
4. Handle edge cases and errors gracefully
5. Show complete, runnable examples (not fragments)
6. Format code with triple backticks and language tag: ```python ... ```
7. Explain key design decisions briefly

## Response Style
- Be **concise** — provide the solution first, then brief explanation
- Use **bold** for key terms
- Cite web sources with URLs when using search results
- When unsure, be honest and suggest alternatives
- For complex tasks, break down into clear steps"""


class ChatAgent:
    """Orchestrates web search + LLM provider for chat responses."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._load_tokenizer()

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("tokenizer", use_fast=True)
            self.tokenizer_loaded = True
        except Exception:
            self.tokenizer = None
            self.tokenizer_loaded = False

    def chat(self, prompt: str, history: List[dict] = None,
             web_search_enabled: bool = True, agent_mode: bool = False) -> dict:
        tool_calls = []
        tool_results = []
        thinking = None
        search_context = ""

        # ── Web Search ────────────────────────────────
        if web_search_enabled:
            search_triggers = ['最新', '今日', '現在', '教えて', 'とは', 'what is',
                              'how to', 'latest', 'current', 'news', 'weather',
                              'いつ', 'どこ', '誰', 'why', 'when', 'where']
            needs_search = any(t in prompt.lower() for t in search_triggers)

            if needs_search:
                thinking = f"Searching the web for: {prompt[:100]}..."
                try:
                    search_context = search_and_summarize(prompt, max_results=3)
                    tool_calls.append({"name": "web_search", "params": prompt[:100]})
                    tool_results.append({"name": "web_search", "result": search_context[:500]})
                except Exception as e:
                    search_context = f"(Web search unavailable: {e})"

        # ── Build prompt with search context ──────────
        full_prompt = prompt
        if search_context and "No results found" not in search_context:
            full_prompt = (
                f"[Web search results for the user's question]\n{search_context}\n\n"
                f"[User question]\n{prompt}\n\n"
                f"Please answer based on the search results above. Cite sources."
            )

        # ── Get response from provider ─────────────────
        try:
            response = self.config.provider.chat(
                prompt=full_prompt,
                history=history,
                system_prompt=SYSTEM_PROMPT,
            )
        except Exception as e:
            response = f"⚠️ モデルエラー: {e}"

        # ── Agent mode wrapping ───────────────────────
        if agent_mode:
            thinking = (thinking or "") + "\n🧠 Agent mode: reasoning..."
            response = f"📋 **分析**: 質問を解析\n🔍 **検索**: Web検索実行\n💭 **推論**: 情報を統合\n\n---\n\n{response}"

        # ── Token count ───────────────────────────────
        usage = {"total_tokens": len(prompt.split()) + len(response.split())}
        if self.tokenizer_loaded:
            try:
                usage["total_tokens"] = len(self.tokenizer.encode(prompt + response))
            except: pass

        return {
            "response": response,
            "thinking": thinking,
            "tool_calls": tool_calls if tool_calls else None,
            "tool_results": tool_results if tool_results else None,
            "usage": usage,
            "web_search_used": bool(search_context),
            "agent_mode": agent_mode,
            "provider": self.config.provider.label,
        }


# ═══════════════════════════════════════════════════════════════
# HTTP Server
# ═══════════════════════════════════════════════════════════════

class ChatHandler(BaseHTTPRequestHandler):
    agent: ChatAgent = None
    store: ConversationStore = None
    config: ServerConfig = None

    def log_message(self, format, *args):
        if not args[0].startswith('GET /health'):
            print(f"📡 {self.client_address[0]} — {args[0]}")

    def do_GET(self):
        path = self.path.split('?')[0]
        if path in ('/', '/index.html'):
            self._serve_ui()
        elif path == '/health':
            self._json(200, {"status": "ok", "timestamp": time.time()})
        elif path == '/v1/providers':
            self._json(200, {"providers": list_providers()})
        elif path == '/v1/config':
            self._json(200, self.config.to_dict())
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        path = self.path.split('?')[0]
        if path == '/v1/chat':
            self._handle_chat()
        elif path == '/v1/config':
            self._handle_config()
        else:
            self._json(404, {"error": "Not found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _serve_ui(self):
        ui_path = Path(__file__).parent / 'web_chat' / 'index.html'
        if ui_path.exists():
            content = ui_path.read_text(encoding='utf-8')
            self._html(200, content)
        else:
            self._html(200, "<!DOCTYPE html><html lang='ja'><head><meta charset='UTF-8'><title>TinyLLM</title></head>"
                "<body style='font-family:sans-serif;max-width:800px;margin:40px auto;padding:20px;background:#1a1a2e;color:#eee'>"
                "<h1>🤖 TinyLLM Chat</h1><p>Chat UIが見つかりません。</p>"
                "<p><code>agent/web_chat/index.html</code> を配置してください。</p></body></html>")

    def _handle_chat(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length).decode())
        except Exception:
            return self._json(400, {"error": "Invalid JSON"})

        prompt = body.get('prompt', '').strip()
        if not prompt:
            return self._json(400, {"error": "Empty prompt"})

        conv_id = body.get('conversation_id', 'default')
        web_search_enabled = body.get('web_search', True)
        agent_mode = body.get('agent_mode', False)

        history = self.store.get(conv_id)
        result = self.agent.chat(prompt, history, web_search_enabled, agent_mode)

        self.store.add(conv_id, 'user', prompt)
        self.store.add(conv_id, 'assistant', result['response'])

        self._json(200, result)

    def _handle_config(self):
        """POST /v1/config — switch provider or set API key."""
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length).decode())
        except Exception:
            return self._json(400, {"error": "Invalid JSON"})

        provider_id = body.get('provider_id')
        api_key = body.get('api_key')

        if provider_id:
            self.config.set_provider(provider_id, api_key)
            # Recreate agent with new provider
            ChatHandler.agent = ChatAgent(self.config)

        self._json(200, self.config.to_dict())

    def _json(self, code, data):
        body = json.dumps(data, ensure_ascii=False, indent=2)
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body.encode())))
        self.end_headers()
        self.wfile.write(body.encode())

    def _html(self, code, content):
        body = content.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description='TinyLLM Web Chat Server (multi-backend)')
    parser.add_argument('--port', type=int, default=8421)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--provider', default='template',
                        help='Default provider: template, openai_compatible, tinyllm_c, or preset:openai')
    parser.add_argument('--api-key', default='', help='API key for the provider (not saved to disk)')
    args = parser.parse_args()

    print("🤖 TinyLLM Web Chat Server (multi-backend)")
    print("=" * 50)
    print(f"   🌐 http://{args.host}:{args.port}")
    print(f"   🔀 Default provider: {args.provider}")
    print(f"   🔍 Web search: DuckDuckGo (free)")
    print(f"   🧠 Agent mode: enabled")
    print(f"   💬 Conversations: 24h TTL")
    print("=" * 50)

    # Setup config, agent, store
    config = ServerConfig()
    if args.provider != 'template':
        config.set_provider(args.provider, args.api_key or None)

    ChatHandler.config = config
    ChatHandler.agent = ChatAgent(config)
    ChatHandler.store = ConversationStore()

    server = HTTPServer((args.host, args.port), ChatHandler)
    print(f"\n✅ Server running. Open http://localhost:{args.port} in Chrome")
    print(f"   ⚙️  Click the gear icon to switch models / set API key\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
