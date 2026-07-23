
import argparse, json
from agent.phase5.autonomous import AutonomousCodingAgent
from agent.phase4.provider import OpenAICompatibleProvider, MockProvider

def main():
    p=argparse.ArgumentParser(description="TinyLLM Phase 5 Autonomous Coding Agent")
    p.add_argument("request")
    p.add_argument("--workspace",default=".")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--test-command",default="python -m pytest -q")
    p.add_argument("--max-attempts",type=int,default=3)
    p.add_argument("--mock",action="store_true")
    args=p.parse_args()
    provider=MockProvider("diff") if args.mock else OpenAICompatibleProvider(base_url=args.base_url,model=args.model)
    result=AutonomousCodingAgent(args.workspace,provider,max_attempts=args.max_attempts).run(args.request,args.test_command)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))

if __name__=="__main__":
    main()
