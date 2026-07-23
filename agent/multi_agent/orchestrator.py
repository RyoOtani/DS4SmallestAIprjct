
from typing import Callable, Dict, Any, List
from .models import AgentRun, AgentTask
from .roles import Planner, Architect, Coder, Tester, Critic, Evaluator, Debugger

class MultiAgentOrchestrator:
    """
    Deterministic orchestration layer for Phase 3.
    LLM calls are injected through callbacks so the system can use local or remote models.
    """

    def __init__(self, llm=None, max_iterations: int = 5):
        self.llm = llm
        self.max_iterations = max_iterations
        self.roles = {
            "planner": Planner(),
            "architect": Architect(),
            "coder": Coder(),
            "tester": Tester(),
            "critic": Critic(),
            "evaluator": Evaluator(),
            "debugger": Debugger(),
        }

    def _call_role(self, role: str, context: Dict[str, Any]):
        result = self.roles[role].run(context)
        context.setdefault("history", []).append({
            "role": result.role,
            "status": result.status,
            "summary": result.summary,
            "data": result.data,
        })
        return result

    def plan(self, request: str) -> AgentRun:
        run = AgentRun(request, max_iterations=self.max_iterations)
        ctx = {"request": request, "history": run.history}
        planned = self._call_role("planner", ctx)
        run.tasks = [AgentTask(**t) for t in planned.data["tasks"]]
        run.status = "planned"
        return run

    def execute(self, run: AgentRun, relevant_files: List[str] = None,
                test_command: str = "python -m pytest -q",
                verify: Callable[[str], Dict[str, Any]] = None) -> AgentRun:
        ctx = {
            "request": run.user_request,
            "history": run.history,
            "relevant_files": relevant_files or [],
            "test_command": test_command,
        }

        self._call_role("architect", ctx)
        self._call_role("coder", ctx)

        for task in run.tasks:
            if task.id in ("inspect", "design", "implement"):
                task.status = "completed"

        for iteration in range(1, self.max_iterations + 1):
            run.iteration = iteration
            verification = verify(run.user_request) if verify else {"ok": True, "failures": []}
            failures = verification.get("failures", []) if not verification.get("ok", True) else []
            ctx["failures"] = failures

            tester = self._call_role("tester", ctx)
            critic = self._call_role("critic", ctx)
            evaluator = self._call_role("evaluator", ctx)

            run.history.append({
                "iteration": iteration,
                "verification": verification,
                "tester": tester.data,
                "critic": critic.data,
                "evaluator": evaluator.data,
            })

            if evaluator.status == "pass":
                run.status = "completed"
                for task in run.tasks:
                    if task.id in ("verify", "review"):
                        task.status = "completed"
                return run

            self._call_role("debugger", ctx)
            run.history.append({"iteration": iteration, "action": "repair_requested"})

        run.status = "failed"
        return run

    def run(self, request: str, **kwargs) -> AgentRun:
        return self.execute(self.plan(request), **kwargs)
