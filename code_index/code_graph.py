"""Repository code graph and hybrid retrieval for TinyLLM Coding Agent v2."""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict

class CodeGraph:
    def __init__(self, index_path="code_index/index.jsonl"):
        self.records=[]
        self.by_symbol=defaultdict(list)
        self.by_path={}
        p=Path(index_path)
        if p.exists():
            for line in p.read_text(encoding="utf8").splitlines():
                if not line.strip(): continue
                r=json.loads(line); self.records.append(r); self.by_path[r["path"]]=r
                for s in r.get("symbols",[]): self.by_symbol[s["name"].lower()].append(r["path"])

    def search(self, query, limit=8):
        terms=[x.lower() for x in re.findall(r"[A-Za-z_][A-Za-z0-9_]+",query) if len(x)>1]
        scored=[]
        for r in self.records:
            hay=(r["path"]+" "+" ".join(s["name"] for s in r.get("symbols",[]))+" "+r.get("text","")).lower()
            score=sum(hay.count(t) for t in terms)
            if score: scored.append((score,r))
        scored.sort(key=lambda x:x[0],reverse=True)
        return [{"score":s,"path":r["path"],"symbols":r.get("symbols",[]),"imports":r.get("imports",[]),
                 "snippet":r.get("text","")[:4000]} for s,r in scored[:limit]]

    def dependencies(self, path):
        r=self.by_path.get(path)
        if not r: return []
        deps=[]
        for imp in r.get("imports",[]):
            token=imp.strip().strip('"<>').split()[0]
            for p in self.by_path:
                if token in p or Path(token).stem in Path(p).stem: deps.append(p)
        return sorted(set(deps))

    def context(self, query, limit=6):
        hits=self.search(query,limit)
        lines=[]
        for h in hits:
            lines.append(f"FILE: {h['path']}\nSYMBOLS: {h['symbols']}\nIMPORTS: {h['imports']}\n{h['snippet']}")
        return "\n\n---\n\n".join(lines)

def build_graph(index_path="code_index/index.jsonl", output="code_index/graph.json"):
    g=CodeGraph(index_path)
    data={"files":list(g.by_path),"symbols":dict(g.by_symbol),
          "dependencies":{p:g.dependencies(p) for p in g.by_path}}
    Path(output).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf8")
    return output
