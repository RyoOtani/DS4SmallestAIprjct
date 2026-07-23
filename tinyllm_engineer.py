
import argparse, json
from agent.phase4.engine import AIEngineer
from agent.phase4.provider import OpenAICompatibleProvider, MockProvider

def main():
    p=argparse.ArgumentParser(description="TinyLLM Phase 4 AI Software Engineer")
    p.add_argument("request")
    p.add_argument("--workspace", default=".")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--test-command", default="python -m pytest -q")
    p.add_argument("--mock", action="store_true")
    args=p.parse_args()
    provider=MockProvider("Phase 4 mock response") if args.mock else OpenAICompatibleProvider(base_url=args.base_url, model=args.model)
    result=AIEngineer(args.workspace, provider=provider).run_cycle(args.request, args.test_command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

if __name__=="__main__":
    main()
