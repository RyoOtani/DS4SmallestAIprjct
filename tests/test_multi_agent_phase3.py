
from agent.multi_agent.orchestrator import MultiAgentOrchestrator

def test_planner_creates_pipeline():
    run = MultiAgentOrchestrator().plan("Add a feature")
    assert run.status == "planned"
    assert [t.id for t in run.tasks] == ["inspect", "design", "implement", "verify", "review"]

def test_orchestrator_passes_with_successful_verification():
    def verify(_):
        return {"ok": True, "failures": []}
    run = MultiAgentOrchestrator(max_iterations=2).run("Test task", verify=verify)
    assert run.status == "completed"
    assert run.iteration == 1

def test_orchestrator_stops_after_max_iterations():
    def verify(_):
        return {"ok": False, "failures": ["test_failure"]}
    run = MultiAgentOrchestrator(max_iterations=2).run("Broken task", verify=verify)
    assert run.status == "failed"
    assert run.iteration == 2
