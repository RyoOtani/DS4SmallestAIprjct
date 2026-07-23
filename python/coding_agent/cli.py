from __future__ import annotations
import argparse, json
from .agent import CodingAgent
from .tools import Toolset
from .workspace import Workspace
from ..runtime.provider import provider_from_env

def main():
    p=argparse.ArgumentParser(description='TinyLLM v2 Coding Agent')
    p.add_argument('task'); p.add_argument('--workspace',default='.')
    p.add_argument('--max-steps',type=int,default=30)
    a=p.parse_args()
    result=CodingAgent(provider_from_env(),Toolset(Workspace(a.workspace)),a.max_steps).run(a.task)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['success'] else 1)
if __name__=='__main__': main()
