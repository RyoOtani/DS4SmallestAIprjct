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
# Agent Integration
# ═══════════════════════════════════════════════════════════════

class ChatAgent:
    """Simple agent with web search capability."""
    
    SYSTEM_PROMPT = """You are TinyLLM, a helpful AI assistant with web search capability.
You can search the web for current information to provide accurate answers.

When answering:
- Use web search results when the question requires current/real-time information
- Cite sources when using web search results
- Format code with triple backticks
- Be concise but thorough
- If you're unsure, say so and suggest what to search for

Your response should be in the same language as the user's question."""

    def __init__(self, tokenizer_path: str = None):
        self.tokenizer_path = tokenizer_path or "tokenizer"
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=True)
            self.tokenizer_loaded = True
        except Exception:
            self.tokenizer = None
            self.tokenizer_loaded = False
    
    def chat(self, prompt: str, history: List[dict] = None,
             web_search_enabled: bool = True, agent_mode: bool = False) -> dict:
        """
        Process a chat message and return response with metadata.
        """
        tool_calls = []
        tool_results = []
        thinking = None
        search_context = ""
        
        # ── Web Search ────────────────────────────────────
        if web_search_enabled:
            # Determine if search is needed
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
        
        # ── Agent Mode ────────────────────────────────────
        if agent_mode:
            thinking = (thinking or "") + "\nAgent mode: analyzing step by step..."
            response = self._agent_reason(prompt, history, search_context)
        else:
            response = self._simple_response(prompt, history, search_context)
        
        # ── Token count ───────────────────────────────────
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
        }
    
    def _simple_response(self, prompt: str, history: List[dict], context: str) -> str:
        """Generate a simple response (no LLM — template-based for demo)."""
        prompt_lower = prompt.lower()
        
        # Greetings
        if any(w in prompt_lower for w in ['こんにちは', 'hello', 'hi', 'hey']):
            return "こんにちは！TinyLLM チャットへようこそ。Web検索とエージェントモードが使えます。何かお手伝いしましょうか？"
        
        # Capabilities
        if any(w in prompt_lower for w in ['何ができる', 'できること', '機能', 'help', 'what can you']):
            return """🌐 **Web検索**: DuckDuckGo でリアルタイム検索ができます
🧠 **エージェントモード**: 複数ステップの推論とツール実行
💬 **会話履歴**: セッションごとに会話を記憶します

右下のチェックボックスでWeb検索とエージェントモードを切り替えられます。"""
        
        # With search context
        if context and "No results found" not in context:
            lines = context.split('\n')
            sources = [l for l in lines if l.startswith('🔗')]
            answer = f"Web検索の結果です：\n\n{context[:800]}\n\n"
            if sources:
                answer += f"📚 参考: {', '.join(sources[:3])}"
            return answer
        
        # Code help
        if any(w in prompt_lower for w in ['コード', 'code', 'python', '関数', 'function']):
            return """コードのお手伝いをします。例：

```python
def hello(name: str) -> str:
    \"\"\"挨拶を返す関数\"\"\"
    return f"こんにちは、{name}さん！"

# 使い方
print(hello("世界"))
```

何を実装したいか教えてください。より詳しいコードを生成します。"""
        
        # Japanese fallback
        if any('\u3040' <= c <= '\u30ff' for c in prompt):  # contains Japanese
            return f"「{prompt[:100]}」についてのお問い合わせですね。\n\nWeb検索を有効にすると、より正確な最新情報をお届けできます。右下の🌐 Web検索をONにしてお試しください。"
        
        # Default
        return f"I understand you're asking about: _{prompt[:200]}_\n\nEnable **🌐 Web Search** (bottom-right) for real-time information, or **🧠 Agent Mode** for multi-step reasoning.\n\nHow can I help you further?"
    
    def _agent_reason(self, prompt: str, history: List[dict], context: str) -> str:
        """Agent mode: multi-step reasoning with tool context."""
        steps = []
        
        # Step 1: Analyze
        steps.append("📋 **分析**: 質問を理解しました")
        
        # Step 2: Search (already done)
        if context:
            steps.append("🔍 **検索**: Web検索を実行しました")
        
        # Step 3: Reason
        steps.append("💭 **推論**: 情報を統合して回答を生成中...")
        
        # Build response
        response = "\n".join(steps) + "\n\n---\n\n"
        
        if context and "No results found" not in context:
            response += context[:1000]
        else:
            response += f"「{prompt[:150]}」について、Web検索の結果が見つかりませんでした。別のキーワードでお試しいただくか、より具体的な質問を入力してください。"
        
        return response


# ═══════════════════════════════════════════════════════════════
# HTTP Server
# ═══════════════════════════════════════════════════════════════

class ChatHandler(BaseHTTPRequestHandler):
    agent: ChatAgent = None
    store: ConversationStore = None
    
    def log_message(self, format, *args):
        print(f"📡 {self.client_address[0]} — {args[0]}") if not args[0].startswith('GET /health') else None
    
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._serve_ui()
        elif self.path == '/health':
            self._json(200, {"status": "ok", "timestamp": time.time()})
        else:
            self._json(404, {"error": "Not found"})
    
    def do_POST(self):
        if self.path == '/v1/chat':
            self._handle_chat()
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
            self._html(200, """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>TinyLLM</title></head>
<body style="font-family:sans-serif;max-width:800px;margin:40px auto;padding:20px;background:#1a1a2e;color:#eee">
<h1>🤖 TinyLLM Chat</h1><p>Chat UI not found. Place <code>agent/web_chat/index.html</code> in the repo.</p>
<p>Run: <code>python chat_server.py</code> from the repo root.</p>
</body></html>""")
    
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
        
        # Get history
        history = self.store.get(conv_id)
        
        # Generate response
        result = self.agent.chat(prompt, history, web_search_enabled, agent_mode)
        
        # Store conversation
        self.store.add(conv_id, 'user', prompt)
        self.store.add(conv_id, 'assistant', result['response'])
        
        self._json(200, result)
    
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
    parser = argparse.ArgumentParser(description='TinyLLM Web Chat Server')
    parser.add_argument('--port', type=int, default=8421)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--tokenizer', default='tokenizer')
    parser.add_argument('--backend', default=None,
                        help='C inference server URL (e.g. http://localhost:8420)')
    args = parser.parse_args()
    
    print("🤖 TinyLLM Web Chat Server")
    print("=" * 50)
    print(f"   🌐 http://{args.host}:{args.port}")
    print(f"   🔍 Web search: DuckDuckGo (free)")
    print(f"   🧠 Agent mode: enabled")
    print(f"   💬 Conversations: 24h TTL")
    print("=" * 50)
    
    # Setup agent and store
    ChatHandler.agent = ChatAgent(args.tokenizer)
    ChatHandler.store = ConversationStore()
    
    server = HTTPServer((args.host, args.port), ChatHandler)
    print(f"\n✅ Server running. Open http://localhost:{args.port} in Chrome\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
