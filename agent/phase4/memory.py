
import json
from pathlib import Path

class AgentMemory:
    def __init__(self, path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.items=[]
        if self.path.exists():
            try: self.items=json.loads(self.path.read_text(encoding="utf-8"))
            except Exception: self.items=[]

    def add(self, kind, content, metadata=None):
        self.items.append({"kind":kind,"content":content,"metadata":metadata or {}})
        self.path.write_text(json.dumps(self.items, ensure_ascii=False, indent=2), encoding="utf-8")

    def recent(self, n=20):
        return self.items[-n:]
