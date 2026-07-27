#!/usr/bin/env python3
"""
Multi-Agent Orchestrator — connects roles to real tool execution.

Runs: Task → Planner → Architect → Coder → Tester → Critic → Evaluator → Done

With LLM: intelligent reasoning at each step.
Without LLM: rule-based fallback with actual tool execution.
"""

from __future__ import annotations
import os, sys, time, subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List

from agent.multi_agent.roles import (
    Planner, Architect, Coder, Tester, Critic, Evaluator, Debugger,
    RoleResult,
)


@dataclass
class PipelineResult:
    task: str
    success: bool
    plan: Optional[RoleResult] = None
    architecture: Optional[RoleResult] = None
    coder_result: Optional[RoleResult] = None
    test_result: Optional[RoleResult] = None
    critic_result: Optional[RoleResult] = None
    evaluator_result: Optional[RoleResult] = None
    duration_s: float = 0.0
    error: str = ""


class MultiAgentOrchestrator:
    """Full pipeline with real tool connections."""

    def __init__(self, provider=None, workspace: str = "."):
        self.provider = provider
        self.workspace = Path(workspace)
        self.tool_results: List[Dict] = []

    def run(self, task: str, files: List[str] = None,
            test_command: str = "python -m pytest -q") -> PipelineResult:
        t0 = time.time()
        result = PipelineResult(task=task, success=False)
        try:
            result.plan = self._run_planner(task, files or [])
            if result.plan.status != "ok":
                result.error = f"Planning failed: {result.plan.summary}"
                result.duration_s = time.time() - t0
                return result
            result.architecture = self._run_architect(task, files or [], result.plan)
            result.coder_result = self._run_coder(task, result.plan, result.architecture)
            result.test_result = self._run_tester(test_command)
            failures = result.test_result.data.get("failures", [])
            result.critic_result = self._run_critic(failures)
            result.evaluator_result = self._run_evaluator(failures)
            result.success = result.evaluator_result.status == "pass"
            result.duration_s = time.time() - t0
        except Exception as e:
            result.error = str(e)
            result.duration_s = time.time() - t0
        return result

    def _run_planner(self, task, files):
        planner = Planner(provider=self.provider)
        # Use improved fallback if no provider
        if not self.provider:
            from agent.multi_agent.orchestrator import improved_fallback_plan
            fb = improved_fallback_plan(task, files)
            return RoleResult("planner", "ok", f"Rule-based plan for: {task}", fb)
        return planner.run({"request": task, "relevant_files": files or self._find_relevant_files(task)})

    def _run_architect(self, task, files, plan):
        arch = Architect(provider=self.provider)
        return arch.run({"request": task, "relevant_files": files, "plan": plan.data})

    def _run_coder(self, task, plan, arch):
        coder = Coder(provider=self.provider)
        result = coder.run({"request": task, "plan": plan.data, "architecture": arch.data})
        for tool_name in result.data.get("suggested_tools", []):
            if tool_name == "read_file":
                self._execute_read_files(arch.data.get("affected_files", []))
            elif tool_name == "write_file":
                self._execute_write_stub(task, plan)
        return result

    def _run_tester(self, test_command):
        tester = Tester(provider=self.provider)
        failures = self._execute_tests(test_command)
        return tester.run({"test_command": test_command, "failures": failures})

    def _run_critic(self, failures):
        return Critic(provider=self.provider).run({"failures": failures})

    def _run_evaluator(self, failures):
        return Evaluator(provider=self.provider).run({"failures": failures})

    def _find_relevant_files(self, task):
        keywords = [w for w in task.lower().split() if len(w) > 2]
        found = []
        for root, dirs, files in os.walk(self.workspace):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('node_modules','__pycache__','venv','.venv')]
            for f in files:
                if f.endswith(('.py','.js','.ts','.c','.h','.rs','.go')):
                    try:
                        with open(os.path.join(root,f),'r',encoding='utf-8',errors='ignore') as fh:
                            if any(kw in fh.read(2000).lower() for kw in keywords):
                                found.append(os.path.join(root,f))
                    except: pass
            if len(found)>=10: break
        return found[:10]

    def _execute_read_files(self, files):
        for fp in files[:5]:
            try:
                with open(fp) as f: size=len(f.read())
                self.tool_results.append({"tool":"read_file","path":fp,"size":size,"status":"ok"})
            except Exception as e:
                self.tool_results.append({"tool":"read_file","path":fp,"error":str(e),"status":"error"})

    def _execute_write_stub(self, task, plan):
        tasks = plan.data.get("tasks",[])
        if not tasks: return
        t = tasks[0]
        content = f"# Auto-generated stub for: {task}\n# Step: {t.get('title','')}\ndef placeholder(): pass\n"
        try:
            with open(f"patch_{t.get('id','fix')}.py",'w') as f: f.write(content)
            self.tool_results.append({"tool":"write_file","status":"ok"})
        except Exception as e:
            self.tool_results.append({"tool":"write_file","error":str(e),"status":"error"})

    def _execute_tests(self, cmd):
        failures = []
        try:
            import shlex
            r = subprocess.run(shlex.split(cmd), shell=False, capture_output=True, text=True, timeout=60, cwd=str(self.workspace))
            if r.returncode != 0:
                for line in (r.stdout+r.stderr).split('\n'):
                    if 'FAILED' in line or 'ERROR' in line: failures.append(line.strip()[:200])
            self.tool_results.append({"tool":"run_tests","exit_code":r.returncode,"failures":len(failures),"status":"ok"})
        except Exception as e:
            failures.append(f"Test error: {e}")
        return failures


def improved_fallback_plan(task: str, files: List[str]) -> dict:
    """Rule-based planning — keyword-aware, better than static templates."""
    tl = task.lower()
    if any(w in tl for w in ['fix','bug','error','crash','broken']):
        return {"tasks":[
            {"id":"reproduce","title":"Reproduce","description":f"Find failing scenario for: {task}"},
            {"id":"diagnose","title":"Diagnose","description":"Check logs and tracebacks."},
            {"id":"patch","title":"Fix","description":"Apply minimal fix."},
            {"id":"verify","title":"Verify","description":"Run tests."}],"strategy":"debug-first"}
    if any(w in tl for w in ['add','implement','create','build','feature']):
        return {"tasks":[
            {"id":"spec","title":"Specify","description":f"Define API contract for: {task}"},
            {"id":"test","title":"Test-first","description":"Write failing tests."},
            {"id":"impl","title":"Implement","description":"Make tests pass."},
            {"id":"integrate","title":"Integrate","description":"Full test suite."}],"strategy":"tdd"}
    if any(w in tl for w in ['refactor','clean','improve','optimize']):
        return {"tasks":[
            {"id":"analyze","title":"Analyze","description":"Map dependencies."},
            {"id":"extract","title":"Extract","description":"Incremental refactor."},
            {"id":"validate","title":"Validate","description":"Full test suite."}],"strategy":"safe-refactor"}
    kw = [w for w in tl.split() if len(w)>3][:3]
    return {"tasks":[
        {"id":"explore","title":"Explore","description":f"Search: {', '.join(kw)}"},
        {"id":"design","title":"Design","description":f"Plan for: {task}"},
        {"id":"implement","title":"Implement","description":"Write and test."},
        {"id":"review","title":"Review","description":"Verify correctness."}],"strategy":"general"}


if __name__ == '__main__':
    print("🧪 MultiAgentOrchestrator")
    orch = MultiAgentOrchestrator()
    r = orch.run("Fix the login bug in auth.py")
    print(f"  Success: {r.success}, plan={len(r.plan.data.get('tasks',[]))} tasks, tools={len(orch.tool_results)}, {r.duration_s:.2f}s")
    assert improved_fallback_plan("Fix crash",[])["strategy"]=="debug-first"
    assert improved_fallback_plan("Add feature",[])["strategy"]=="tdd"
    assert improved_fallback_plan("Refactor db",[])["strategy"]=="safe-refactor"
    print("✅ MultiAgentOrchestrator test passed")
