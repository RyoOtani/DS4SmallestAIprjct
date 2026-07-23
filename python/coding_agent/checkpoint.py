"""Git checkpoint/rollback helper for safe autonomous coding."""
from __future__ import annotations
import subprocess, time

class GitCheckpoint:
    def __init__(self, root): self.root=str(root); self.last=None
    def _run(self,args):
        return subprocess.run(args,cwd=self.root,text=True,capture_output=True)
    def create(self,label="tinyllm-agent"):
        self._run(["git","add","-A"])
        msg=f"{label} checkpoint {int(time.time())}"
        p=self._run(["git","commit","-m",msg,"--no-verify"])
        if p.returncode==0:
            self.last=self._run(["git","rev-parse","HEAD"]).stdout.strip()
            return {"ok":True,"commit":self.last,"message":msg}
        return {"ok":False,"stderr":p.stderr[-4000:]}
    def rollback(self, commit=None):
        target=commit or self.last
        if not target: return {"ok":False,"error":"no checkpoint"}
        p=self._run(["git","reset","--hard",target])
        return {"ok":p.returncode==0,"stdout":p.stdout[-2000:],"stderr":p.stderr[-2000:]}
