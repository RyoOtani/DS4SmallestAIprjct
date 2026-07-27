#!/usr/bin/env python3
"""
web_search.py — DuckDuckGo / Google web search tool for TinyLLM agent.

Provides:
  - web_search(query) → list of results (title, url, snippet)
  - fetch_page(url) → page text content
  - Cached results to avoid repeated API calls

Usage:
  from agent.web_search import web_search, fetch_page
  results = web_search("Python async tutorial")
  content = fetch_page(results[0]['url'])
"""

import json, os, time, hashlib
from pathlib import Path
from typing import List, Dict, Optional
from urllib.request import Request, urlopen
from urllib.parse import quote_plus
from urllib.error import URLError

# Cache directory
CACHE_DIR = Path(os.environ.get("TINYLLM_CACHE_DIR", "data/search_cache"))
CACHE_TTL = 86400  # 24 hours


def _cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()[:16]


def _cache_get(query: str) -> Optional[List[Dict]]:
    """Read cached search results."""
    cache_file = CACHE_DIR / f"{_cache_key(query)}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as f:
            data = json.load(f)
        if time.time() - data.get("timestamp", 0) < CACHE_TTL:
            return data.get("results")
    except Exception:
        pass
    return None


def _cache_set(query: str, results: List[Dict]):
    """Write search results to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{_cache_key(query)}.json"
    try:
        with open(cache_file, 'w') as f:
            json.dump({"timestamp": time.time(), "results": results}, f)
    except Exception:
        pass


def web_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search the web using DuckDuckGo.
    
    Uses `duckduckgo_search` library if available (best results),
    falls back to DDG Instant Answer API, then DDG HTML lite.
    
    Returns list of {title, url, snippet}.
    """
    if not query.strip():
        return []
    
    # Check cache first
    cached = _cache_get(query)
    if cached:
        return cached[:max_results]
    
    results = []
    
    # ── Method 1: ddgs library (best) ─────────────────
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")[:300],
                })
        if results:
            _cache_set(query, results)
            return results[:max_results]
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  ddgs search error: {e}")
    
    # ── Method 2: DDG Instant Answer API ─────────────────
    try:
        url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
        req = Request(url, headers={"User-Agent": "TinyLLM/1.0"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        
        if data.get("AbstractText"):
            results.append({
                "title": data.get("AbstractSource", "DuckDuckGo"),
                "url": data.get("AbstractURL", ""),
                "snippet": data["AbstractText"],
            })
        
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic["Text"],
                })
    except Exception as e:
        print(f"⚠️  DDG API error: {e}")
    
    # ── Method 3: DDG HTML lite scraping ─────────────────
    if not results:
        results = _html_search(query, max_results)
    
    # ── Last resort ─────────────────────────────────────
    if not results:
        results = [{
            "title": f"Search: {query}",
            "url": f"https://www.google.com/search?q={quote_plus(query)}",
            "snippet": f"Web search results for '{query}' are unavailable. "
                       f"Check your internet connection or try again later.",
        }]
    
    if results:
        _cache_set(query, results)
    return results[:max_results]


def _html_search(query: str, max_results: int = 5) -> List[Dict]:
    """
    DuckDuckGo HTML search fallback (lite version).
    Parses the lite.duckduckgo.com results page.
    """
    results = []
    try:
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; TinyLLM/1.0)"})
        with urlopen(req, timeout=10) as resp:
            html = resp.read().decode('utf-8', errors='replace')
        
        # Parse lite DDG results — each result is a <a> with class "result-link"
        import re
        # Find result rows: <a rel="nofollow" class="result-link" href="URL">TITLE</a>
        # followed by <span class="result-snippet">SNIPPET</span>
        links = re.findall(r'<a[^>]*result-link[^>]*href="([^"]*)"[^>]*>(.*?)</a>', html, re.DOTALL)
        snippets = re.findall(r'result-snippet[^>]*>(.*?)</span>', html, re.DOTALL)
        
        for i, (url, title) in enumerate(links[:max_results]):
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
            if title and url:
                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet[:300],
                })
    except Exception as e:
        print(f"⚠️  HTML search error: {e}")
    
    return results


def _fallback_search(query: str, max_results: int = 5) -> List[Dict]:
    """Fallback chain: DDG API → DDG HTML lite → Google URL."""
    # Try DDG HTML lite
    results = _html_search(query, max_results)
    if results:
        return results
    
    # Last resort: Google search URL
    return [{
        "title": f"Search: {query}",
        "url": f"https://www.google.com/search?q={quote_plus(query)}",
        "snippet": f"Web search results for '{query}' are unavailable offline. "
                   f"Check your internet connection or try again later.",
    }]


def fetch_page(url: str, timeout: int = 10) -> Optional[str]:
    """
    Fetch and extract text content from a web page.
    Strips HTML tags for plain text output.
    """
    if not url:
        return None
    
    try:
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (TinyLLM/1.0; compatible; +https://github.com/RyoOtani/DS4SmallestAIprjct)"
        })
        with urlopen(req, timeout=timeout) as resp:
            html = resp.read().decode('utf-8', errors='replace')
        
        # Simple HTML → text extraction (no external deps)
        import re
        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Limit length
        return text[:10000] if text else None
    except Exception as e:
        return f"[Error fetching {url}: {e}]"


def search_and_summarize(query: str, max_results: int = 3) -> str:
    """
    Search the web and return a formatted summary for the agent.
    This is the primary function the agent should call.
    """
    results = web_search(query, max_results)
    if not results:
        return f"No results found for: {query}"
    
    lines = [f"Web search results for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r['title']}**")
        lines.append(f"   {r['snippet'][:300]}")
        if r.get('url'):
            lines.append(f"   🔗 {r['url']}")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == '__main__':
    print("🌐 TinyLLM Web Search Tool")
    print("=" * 50)
    
    # Test search
    results = web_search("Python asyncio tutorial")
    print(f"Results: {len(results)}")
    for r in results:
        print(f"  📄 {r['title']}")
        print(f"     {r['snippet'][:100]}...")
    
    print(f"\n📋 Summary:\n{search_and_summarize('What is a transformer model?', 2)}")
