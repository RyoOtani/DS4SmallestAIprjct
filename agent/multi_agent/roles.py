
"""
Multi-Agent Role System with optional LLM integration.

When a provider is available, roles use it for intelligent reasoning.
Without a provider, roles fall back to template-based responses.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class RoleResult:
    role: str
    status: str
    summary: str
    data: Dict[str, Any]


class BaseRole:
    name = "base"

    def __init__(self, provider=None):
        """provider: optional LLMProvider for intelligent reasoning."""
        self.provider = provider

    def run(self, context: Dict[str, Any]) -> RoleResult:
        raise NotImplementedError

    def _ask_llm(self, system: str, prompt: str) -> Optional[str]:
        """Query LLM if provider is available."""
        if not self.provider:
            return None
        try:
            return self.provider.complete([
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            return None


class Planner(BaseRole):
    name = "planner"
    def run(self, context):
        request = context.get("request", "")
        llm_response = self._ask_llm(
            "You are a senior software project planner. Break down tasks into concrete steps.",
            f"Plan the implementation for: {request}\n\nContext files: {context.get('relevant_files', [])}\n\nOutput a list of 3-6 concrete tasks with IDs, titles, and descriptions."
        )
        if llm_response:
            # Parse LLM response into tasks
            tasks = self._parse_tasks(llm_response)
            if tasks:
                return RoleResult(self.name, "ok", f"LLM-planned: {request}", {"tasks": tasks})
        
        # Fallback: template-based
        tasks = [
            {"id": "inspect", "title": "Inspect repository", "description": "Index and inspect relevant files."},
            {"id": "design", "title": "Design implementation", "description": "Define the architecture and affected symbols."},
            {"id": "implement", "title": "Implement changes", "description": "Modify the smallest safe set of files."},
            {"id": "verify", "title": "Verify changes", "description": "Build and run relevant tests."},
            {"id": "review", "title": "Review changes", "description": "Critique correctness, regressions, and security."},
        ]
        return RoleResult(self.name, "ok", f"Created plan for: {request}", {"tasks": tasks})

    def _parse_tasks(self, response: str) -> list:
        """Parse structured task list from LLM output."""
        tasks = []
        for line in response.split('\n'):
            line = line.strip()
            if ':' in line and any(line.lower().startswith(p) for p in ['task', 'step', '-', '*', '1', '2', '3', '4', '5', '6']):
                parts = line.split(':', 1)
                title = parts[1].strip() if len(parts) > 1 else line
                tasks.append({
                    "id": f"task_{len(tasks)+1}",
                    "title": title[:80],
                    "description": title,
                })
        return tasks[:6] if tasks else []


class Architect(BaseRole):
    name = "architect"
    def run(self, context):
        request = context.get("request", "")
        files = context.get("relevant_files", [])
        
        llm_response = self._ask_llm(
            "You are a software architect. Design implementation strategies.",
            f"Task: {request}\nFiles: {files}\nOutput: affected files list and constraints."
        )
        if llm_response:
            return RoleResult(self.name, "ok", "LLM-designed architecture.",
                            {"affected_files": files, "design_notes": llm_response[:500]})
        
        return RoleResult(
            self.name, "ok",
            "Produced an implementation brief from repository context.",
            {"affected_files": files,
             "constraints": ["preserve public APIs where possible", "prefer minimal diffs", "add tests for behavior changes"]}
        )


class Coder(BaseRole):
    name = "coder"
    def run(self, context):
        request = context.get("request", "")
        plan = context.get("plan", {})
        
        llm_response = self._ask_llm(
            "You are a coding assistant. Suggest which tools to use for a task.",
            f"Task: {request}\nPlan: {plan}\nAvailable tools: read_file, write_file, run_command, search_code, find_symbol\nRespond with tool names only."
        )
        tools = ["read_file", "write_file", "run_command"]
        if llm_response:
            suggested = [t for t in tools if t in llm_response.lower()]
            if suggested:
                tools = suggested
        
        return RoleResult(
            self.name, "ready" if llm_response else "ready",
            "Coder ready — " + ("LLM-suggested tools." if llm_response else "using default tools."),
            {"action": "use_tools", "suggested_tools": tools,
             "llm_used": llm_response is not None}
        )


class Tester(BaseRole):
    name = "tester"
    def run(self, context):
        test_cmd = context.get("test_command", "python -m pytest -q")
        failures = context.get("failures", [])
        
        llm_response = self._ask_llm(
            "You are a QA engineer. Suggest test commands for verification.",
            f"Test command: {test_cmd}\nFailures: {failures}\nSuggest improvements or alternative test commands."
        )
        if llm_response:
            return RoleResult(self.name, "ready", "LLM-suggested test strategy.",
                            {"action": "run_tests", "command": test_cmd, "suggestion": llm_response[:300]})
        
        return RoleResult(
            self.name, "ready",
            "Tester is ready to execute the configured verification command.",
            {"action": "run_tests", "command": test_cmd}
        )


class Critic(BaseRole):
    name = "critic"
    def run(self, context):
        failures = context.get("failures", [])
        
        llm_response = self._ask_llm(
            "You are a code reviewer. Analyze test failures and suggest fixes.",
            f"Failures: {failures}\nOutput: root cause analysis and fix suggestions."
        ) if failures else None
        
        if llm_response:
            return RoleResult(self.name, "needs_fix" if failures else "ok",
                            f"LLM analysis: {llm_response[:100]}",
                            {"failures": failures, "analysis": llm_response[:500]})
        
        return RoleResult(
            self.name, "ok" if not failures else "needs_fix",
            "No known verification failures." if not failures else "Verification failures require debugging.",
            {"failures": failures}
        )


class Evaluator(BaseRole):
    name = "evaluator"
    def run(self, context):
        failures = context.get("failures", [])
        criteria = context.get("criteria", ["build", "tests", "no critical regression"])
        
        llm_response = self._ask_llm(
            "You are a build engineer. Evaluate if changes pass all criteria.",
            f"Failures: {failures}\nCriteria: {criteria}\nRespond: PASS or FAIL with reason."
        ) if failures else None
        
        passed = not failures
        summary = "PASS" if passed else "FAIL"
        if llm_response:
            passed = "PASS" in llm_response.upper()
            summary = llm_response[:100]
        
        return RoleResult(
            self.name, "pass" if passed else "fail",
            summary,
            {"pass": passed, "criteria": criteria, "llm_used": llm_response is not None}
        )


class Debugger(BaseRole):
    name = "debugger"
    def run(self, context):
        failures = context.get("failures", [])
        
        llm_response = self._ask_llm(
            "You are a debugging expert. Diagnose test failures and propose fixes.",
            f"Failures: {failures}\nOutput: diagnosis and repair steps."
        ) if failures else None
        
        if llm_response:
            return RoleResult(self.name, "ready", f"LLM diagnosis: {llm_response[:100]}",
                            {"action": "repair", "failures": failures, "diagnosis": llm_response[:500]})
        
        return RoleResult(
            self.name, "ready",
            "Debugger prepared a repair cycle from the latest failure report.",
            {"action": "repair", "failures": failures}
        )
