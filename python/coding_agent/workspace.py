"""Workspace-safe filesystem operations for the coding agent."""
from __future__ import annotations
import os
from pathlib import Path

IGNORED = {'.git', '.venv', '__pycache__', 'node_modules', 'build', 'dist', '.DS_Store'}

class Workspace:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        if not self.root.exists(): raise FileNotFoundError(self.root)

    def resolve(self, path: str) -> Path:
        p = (self.root / path).resolve() if not os.path.isabs(path) else Path(path).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError(f"path escapes workspace: {path}")
        return p

    def list_files(self, max_files: int = 5000):
        out=[]
        for p in self.root.rglob('*'):
            if any(part in IGNORED for part in p.parts): continue
            if p.is_file():
                out.append(str(p.relative_to(self.root)))
                if len(out)>=max_files: break
        return out

    def read(self, path: str, max_bytes: int = 200_000):
        p=self.resolve(path); return p.read_text(errors='replace')[:max_bytes]

    def write(self, path: str, content: str):
        p=self.resolve(path); p.parent.mkdir(parents=True, exist_ok=True); p.write_text(content)
        return str(p.relative_to(self.root))

    def search(self, query: str, max_hits: int = 100):
        hits=[]
        q=query.lower()
        for rel in self.list_files():
            p=self.root/rel
            try: text=p.read_text(errors='replace')
            except: continue
            for i,line in enumerate(text.splitlines(),1):
                if q in line.lower():
                    hits.append(f"{rel}:{i}:{line[:300]}")
                    if len(hits)>=max_hits: return hits
        return hits
