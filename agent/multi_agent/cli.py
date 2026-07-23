
import argparse, json
from .orchestrator import MultiAgentOrchestrator

def main():
    p = argparse.ArgumentParser(description="TinyLLM Phase 3 Multi-Agent orchestrator")
    p.add_argument("request")
    p.add_argument("--max-iterations", type=int, default=5)
    args = p.parse_args()

    run = MultiAgentOrchestrator(max_iterations=args.max_iterations).plan(args.request)
    print(json.dumps({
        "status": run.status,
        "request": run.user_request,
        "tasks": [t.__dict__ for t in run.tasks],
    }, indent=2))

if __name__ == "__main__":
    main()
