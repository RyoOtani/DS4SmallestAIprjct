
import subprocess

class GitCheckpoint:
    def __init__(self, cwd):
        self.cwd=cwd

    def create(self, message="tinyllm-agent-checkpoint"):
        p=subprocess.run(["git","add","-A"],cwd=self.cwd,capture_output=True,text=True)
        if p.returncode: return {"ok":False,"error":p.stderr}
        p=subprocess.run(["git","commit","-m",message],cwd=self.cwd,capture_output=True,text=True)
        return {"ok":p.returncode==0,"stdout":p.stdout,"stderr":p.stderr}

    def rollback(self, commit="HEAD"):
        p=subprocess.run(["git","reset","--hard",commit],cwd=self.cwd,capture_output=True,text=True)
        return {"ok":p.returncode==0,"stdout":p.stdout,"stderr":p.stderr}
