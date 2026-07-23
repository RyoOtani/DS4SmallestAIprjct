
"""TinyLLM Phase 3 entry point."""
from agent.multi_agent.orchestrator import MultiAgentOrchestrator

def create_orchestrator(max_iterations=5):
    return MultiAgentOrchestrator(max_iterations=max_iterations)

if __name__ == "__main__":
    import sys
    request = " ".join(sys.argv[1:]) or "Inspect the repository and propose a safe implementation plan."
    run = create_orchestrator().run(request)
    print(run.status)
