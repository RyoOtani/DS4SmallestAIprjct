
from pathlib import Path
import subprocess

class WorkspaceTools:
    """Safe workspace-scoped tool layer for coding agents."""
    def __init__(self, workspace, timeout=60):
        self.root = Path(workspace).resolve()
        self.timeout = timeout

    def _safe(self, rel):
        p=(self.root/rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError("Path escapes workspace")
        return p

    def read_file(self, path):
        return self._safe(path).read_text(encoding="utf-8")

    def write_file(self, path, content):
        p=self._safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return str(p.relative_to(self.root))

    def search(self, query, glob="*"):
        hits=[]
        for p in self.root.rglob(glob):
            if p.is_file():
                try:
                    txt=p.read_text(encoding="utf-8")
                except Exception:
                    continue
                if query in txt:
                    hits.append(str(p.relative_to(self.root)))
        return hits[:100]

    def run(self, command):
        proc=subprocess.run(command, cwd=self.root, shell=True, capture_output=True,
                            text=True, timeout=self.timeout)
        return {"returncode":proc.returncode,"stdout":proc.stdout,"stderr":proc.stderr}

    def git_diff(self):
        return self.run("git diff -- .")
