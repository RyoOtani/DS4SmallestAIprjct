"""Phase 2 autonomous coding loop: inspect -> plan -> implement -> verify -> critique -> repair."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from .workspace import Workspace
from .tools import Toolset
from .agent import CodingAgent
from .review import ReviewPipeline
from ..runtime.provider import Provider

def run_phase2(provider, workspace, task, max_steps=30):
    ws=Workspace(workspace); tools=Toolset(ws)
    # Index if available
    idx=Path(workspace)/"code_index/index.jsonl"
    if not idx.exists():
        import subprocess
        subprocess.run(["python","-m","code_index.indexer",workspace,"-o",str(idx)],capture_output=True,text=True)
        tools.graph = tools.graph.__class__(str(idx)) if tools.graph else None
    checkpoint=tools.execute("git_checkpoint",{"label":"phase2-start"})
    agent=CodingAgent(provider,tools,max_steps=max_steps)
    result=agent.run(task)
    review=ReviewPipeline(provider,tools,tools.graph).review(task)
    return {"agent":result,"review":review,"checkpoint":checkpoint}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workspace"); p.add_argument("task")
    p.add_argument("--max-steps",type=int,default=30)
    p.add_argument("--provider",default="openai_compatible")
    a=p.parse_args()
    from ..runtime.provider import OpenAICompatibleProvider, LocalCLIProvider
    provider=OpenAICompatibleProvider() if a.provider=="openai_compatible" else LocalCLIProvider()
    print(json.dumps(run_phase2(provider,a.workspace,a.task,a.max_steps),ensure_ascii=False,indent=2))

if __name__=="__main__": main()
