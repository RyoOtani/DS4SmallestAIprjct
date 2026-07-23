"""Tool registry for the coding agent. Every tool is workspace-scoped."""
from __future__ import annotations
import json, subprocess, time
from .workspace import Workspace
from .checkpoint import GitCheckpoint
try:
    from ...code_index.code_graph import CodeGraph
except Exception:
    CodeGraph = None

class Toolset:
    def __init__(self, workspace: Workspace, sandbox=None):
        self.ws=workspace; self.sandbox=sandbox
        self.checkpoint=GitCheckpoint(self.ws.root)
        self.graph=CodeGraph(str(self.ws.root/"code_index/index.jsonl")) if CodeGraph else None

    def execute(self, name, args):
        if name=='list_files': return {'files': self.ws.list_files()}
        if name=='read_file': return {'path':args['path'],'content':self.ws.read(args['path'])}
        if name=='search_code': return {'hits':self.ws.search(args['query'])}
        if name=='write_file': return {'written':self.ws.write(args['path'],args['content'])}
        if name=='run_command':
            cmd=args['command']; cwd=str(self.ws.root)
            p=subprocess.run(cmd,shell=True,cwd=cwd,text=True,capture_output=True,timeout=args.get('timeout',60))
            return {'returncode':p.returncode,'stdout':p.stdout[-10000:],'stderr':p.stderr[-10000:]}
        if name=='code_context':
            if not self.graph: return {'context':''}
            return {'context':self.graph.context(args['query'],args.get('limit',6))}
        if name=='git_checkpoint': return self.checkpoint.create(args.get('label','agent'))
        if name=='git_rollback': return self.checkpoint.rollback(args.get('commit'))
        if name=='git_diff':
            p=subprocess.run(['git','diff','--'],cwd=self.ws.root,text=True,capture_output=True)
            return {'returncode':p.returncode,'diff':p.stdout[-30000:]}
        if name=='run_tests':
            cmd=args.get('command','pytest -q')
            p=subprocess.run(cmd,shell=True,cwd=self.ws.root,text=True,capture_output=True,timeout=args.get('timeout',120))
            return {'returncode':p.returncode,'stdout':p.stdout[-12000:],'stderr':p.stderr[-12000:]}
        raise ValueError(f'unknown tool: {name}')

    def definitions(self):
        return [
          {'name':'list_files','description':'List workspace files','parameters':{}},
          {'name':'read_file','description':'Read a workspace file','parameters':{'path':'string'}},
          {'name':'search_code','description':'Search source code','parameters':{'query':'string'}},
          {'name':'write_file','description':'Write or replace a workspace file','parameters':{'path':'string','content':'string'}},
          {'name':'run_command','description':'Run a build or development command in the workspace','parameters':{'command':'string','timeout':'integer'}},
          {'name':'run_tests','description':'Run tests and return output','parameters':{'command':'string','timeout':'integer'}},
          {'name':'git_diff','description':'Show current git diff','parameters':{}},
          {'name':'code_context','description':'Retrieve relevant repository code using the code index','parameters':{'query':'string','limit':'integer'}},
          {'name':'git_checkpoint','description':'Create a git checkpoint before risky work','parameters':{'label':'string'}},
          {'name':'git_rollback','description':'Rollback to the latest or specified checkpoint','parameters':{'commit':'string'}},
        ]
