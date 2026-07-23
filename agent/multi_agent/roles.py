
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class RoleResult:
    role: str
    status: str
    summary: str
    data: Dict[str, Any]

class BaseRole:
    name = "base"

    def run(self, context: Dict[str, Any]) -> RoleResult:
        raise NotImplementedError

class Planner(BaseRole):
    name = "planner"
    def run(self, context):
        request = context["request"]
        tasks = [
            {"id": "inspect", "title": "Inspect repository", "description": "Index and inspect relevant files."},
            {"id": "design", "title": "Design implementation", "description": "Define the architecture and affected symbols."},
            {"id": "implement", "title": "Implement changes", "description": "Modify the smallest safe set of files."},
            {"id": "verify", "title": "Verify changes", "description": "Build and run relevant tests."},
            {"id": "review", "title": "Review changes", "description": "Critique correctness, regressions, and security."},
        ]
        return RoleResult(self.name, "ok", f"Created plan for: {request}", {"tasks": tasks})

class Architect(BaseRole):
    name = "architect"
    def run(self, context):
        return RoleResult(
            self.name, "ok",
            "Produced an implementation brief from repository context.",
            {"affected_files": context.get("relevant_files", []),
             "constraints": ["preserve public APIs where possible", "prefer minimal diffs", "add tests for behavior changes"]}
        )

class Coder(BaseRole):
    name = "coder"
    def run(self, context):
        return RoleResult(
            self.name, "ready",
            "Coder is ready to apply changes through the existing tool layer.",
            {"action": "use_tools", "suggested_tools": ["read_file", "write_file", "run_command"]}
        )

class Tester(BaseRole):
    name = "tester"
    def run(self, context):
        return RoleResult(
            self.name, "ready",
            "Tester is ready to execute the configured verification command.",
            {"action": "run_tests", "command": context.get("test_command", "python -m pytest -q")}
        )

class Critic(BaseRole):
    name = "critic"
    def run(self, context):
        failures = context.get("failures", [])
        return RoleResult(
            self.name, "ok" if not failures else "needs_fix",
            "No known verification failures." if not failures else "Verification failures require debugging.",
            {"failures": failures}
        )

class Evaluator(BaseRole):
    name = "evaluator"
    def run(self, context):
        failures = context.get("failures", [])
        return RoleResult(
            self.name, "pass" if not failures else "fail",
            "PASS" if not failures else "FAIL",
            {"pass": not failures, "criteria": ["build", "tests", "no critical regression"]}
        )

class Debugger(BaseRole):
    name = "debugger"
    def run(self, context):
        return RoleResult(
            self.name, "ready",
            "Debugger prepared a repair cycle from the latest failure report.",
            {"action": "repair", "failures": context.get("failures", [])}
        )
