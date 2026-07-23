"""
Phase 6 Orchestrator — End-to-end AI Software Engineer Professional pipeline.

Integrates:
  - RepositoryAnalyzer (deep code understanding)
  - ArchitectAgent (architecture analysis)
  - ToolRegistry + ToolCallParser (structured tool calling)
  - QualityPipeline (lint/format/typecheck/static analysis)
  - CriticAgent + DebuggerAgent (review & debug)
  - Phase 5 autonomous coding loop

The orchestrator runs the full professional-grade development cycle:
  Understand → Plan → Implement → Build → Test → Review → Fix → Done

v2.1: Added agent synchronization via file locks and concurrent access protection.
"""

from __future__ import annotations
import fcntl
import json
import os
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Set

from .code_understanding import RepositoryAnalyzer
from .architect import ArchitectAgent
from .tool_calling import ToolRegistry, ToolCallParser
from .quality import QualityPipeline
from .critic import CriticAgent, DebuggerAgent, CodeReview
from ..phase4.provider import LLMProvider, MockProvider


class FileLockManager:
    """Prevents concurrent modification of files by multiple agents."""

    def __init__(self, lock_dir: Optional[str] = None):
        self.lock_dir = Path(lock_dir or ".tinyllm_locks")
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self._held_locks: Set[str] = set()
        self._thread_lock = threading.Lock()

    def _lock_path(self, filepath: str) -> Path:
        """Generate a lock file path from the target file path."""
        safe_name = filepath.replace("/", "_").replace("\\", "_")
        return self.lock_dir / f"{safe_name}.lock"

    @contextmanager
    def lock_file(self, filepath: str, timeout: float = 30.0):
        """Context manager to exclusively lock a file for editing."""
        lock_file = self._lock_path(filepath)
        start = time.time()

        with self._thread_lock:
            if filepath in self._held_locks:
                raise RuntimeError(f"File already locked by this orchestrator: {filepath}")

            # Wait for lock with timeout
            while lock_file.exists():
                if time.time() - start > timeout:
                    raise TimeoutError(f"Timed out waiting for file lock: {filepath}")
                time.sleep(0.1)

            # Acquire lock
            lock_file.write_text(str(os.getpid()))
            self._held_locks.add(filepath)

        try:
            yield
        finally:
            with self._thread_lock:
                if lock_file.exists():
                    lock_file.unlink()
                self._held_locks.discard(filepath)

    def is_locked(self, filepath: str) -> bool:
        return self._lock_path(filepath).exists()


class Phase6Orchestrator:
    """Orchestrates the full Phase 6 AI Software Engineer pipeline."""

    def __init__(
        self,
        workspace: str,
        provider: Optional[LLMProvider] = None,
        max_attempts: int = 5,
    ):
        self.workspace = Path(workspace).resolve()
        self.provider = provider or MockProvider("OK")
        self.max_attempts = max_attempts

        # Core modules
        self.analyzer = RepositoryAnalyzer(str(workspace))
        self.architect = None
        self.tool_registry = ToolRegistry()
        self.quality = QualityPipeline()
        self.critic = CriticAgent()
        self.debugger = DebuggerAgent()

        # Synchronization
        self.file_locks = FileLockManager()
        self._agent_states: dict[str, str] = {}
        self._agent_state_lock = threading.Lock()

    def set_agent_state(self, agent_name: str, state: str):
        """Record an agent's state for cross-agent coordination."""
        with self._agent_state_lock:
            self._agent_states[agent_name] = state

    def get_agent_states(self) -> dict[str, str]:
        """Get all agent states (for planning/conflict resolution)."""
        with self._agent_state_lock:
            return dict(self._agent_states)

    def edit_file_safe(self, filepath: str, new_content: str) -> bool:
        """Safely edit a file with locking."""
        target = (self.workspace / filepath).resolve()
        if target != self.workspace and self.workspace not in target.parents:
            return False  # workspace escape prevention

        with self.file_locks.lock_file(str(target)):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_content, encoding="utf-8")
        return True
        
        # State
        self.history: list[dict] = []
        self.checkpoints: list[str] = []
    
    def scan_repository(self) -> dict:
        """Step 1: Deep code understanding — scan entire repo."""
        self.analyzer.scan()
        self.architect = ArchitectAgent(self.analyzer, self.provider)
        
        return {
            "files": self.analyzer.metrics.get("total_files", 0),
            "symbols": self.analyzer.metrics.get("total_symbols", 0),
            "functions": self.analyzer.metrics.get("total_functions", 0),
            "classes": self.analyzer.metrics.get("total_classes", 0),
        }
    
    def analyze_architecture(self) -> dict:
        """Step 2: Architecture analysis."""
        if not self.architect:
            self.scan_repository()
        
        arch = self.architect.analyze_current_architecture()
        suggestions = self.architect.suggest_improvements()
        
        return {
            "components": len(arch.components),
            "patterns": arch.patterns,
            "risks": arch.risks,
            "suggestions": suggestions,
        }
    
    def understand_file(self, filepath: str) -> str:
        """Get deep understanding of a specific file."""
        return self.analyzer.context_for_file(filepath)
    
    def find_symbol(self, name: str) -> dict:
        """Find a symbol across the entire repository."""
        symbols = self.analyzer.find_symbol_usages(name)
        callers = self.analyzer.find_callers(name)
        callees = self.analyzer.find_callees(name)
        return {
            "definitions": [
                {"name": s.name, "kind": s.kind, "file": s.file, "line": s.line,
                 "signature": s.signature, "docstring": s.docstring[:200]}
                for s in symbols
            ],
            "callers": [{"caller": e.caller, "file": e.caller_file} for e in callers[:20]],
            "called_by_you": [{"callee": e.callee, "file": e.callee_file} for e in callees[:20]],
        }
    
    def run_quality_pipeline(self, auto_fix: bool = False) -> dict:
        """Step 3: Quality assurance — lint, format, type check, static analysis."""
        return self.quality.run_full_pipeline(str(self.workspace), auto_fix=auto_fix)
    
    def review_code(self, filepath: Optional[str] = None) -> dict:
        """Step 4: Code review — critic analysis."""
        if filepath:
            review = self.critic.review_file(filepath)
            return self._review_to_dict(review)
        else:
            reviews = self.critic.review_directory(str(self.workspace))
            total_score = sum(r.score for r in reviews.values()) / max(len(reviews), 1)
            return {
                "files_reviewed": len(reviews),
                "average_score": round(total_score, 1),
                "blockers": sum(len(r.blockers) for r in reviews.values()),
                "details": {f: self._review_to_dict(r) for f, r in reviews.items()},
            }
    
    def _review_to_dict(self, review: CodeReview) -> dict:
        return {
            "score": review.score,
            "passed": review.passed,
            "blockers": len(review.blockers),
            "total_comments": len(review.comments),
            "blocker_details": [
                {"file": c.file, "line": c.line, "category": c.category, "message": c.message}
                for c in review.blockers[:10]
            ],
        }
    
    def build_and_test(self, build_cmd: str = "", test_cmd: str = "") -> dict:
        """Step 5: Build & test with debugger analysis."""
        result = {"build": None, "test": None}
        
        if build_cmd:
            build_result = self.debugger.run_build(build_cmd, str(self.workspace))
            result["build"] = {
                "status": build_result.status,
                "diagnosis": build_result.diagnosis,
                "error_file": build_result.error_file,
                "error_line": build_result.error_line,
                "suggestions": build_result.suggestions,
            }
        
        if test_cmd:
            test_result = self.debugger.run_tests(test_cmd, str(self.workspace))
            result["test"] = {
                "status": test_result.status,
                "diagnosis": test_result.diagnosis,
                "error_file": test_result.error_file,
                "error_line": test_result.error_line,
                "stack_trace": test_result.stack_trace,
                "suggestions": test_result.suggestions,
            }
        
        return result
    
    def plan_task(self, task: str) -> str:
        """Use LLM to plan a task with full repository context."""
        scan = self.scan_repository()
        arch = self.analyze_architecture()
        
        context = {
            "repository": scan,
            "architecture": arch,
            "top_files": self.analyzer.metrics.get("most_complex_files", [])[:5],
        }
        
        prompt = (
            "You are an AI Software Engineer. Plan a detailed implementation for the task below.\n"
            "Consider the repository structure, existing patterns, and dependencies.\n\n"
            f"## Repository Context\n```json\n{json.dumps(context, indent=2, ensure_ascii=False)}\n```\n\n"
            f"## Task\n{task}\n\n"
            "## Plan\nProvide a step-by-step plan including:\n"
            "1. Files to create/modify\n"
            "2. Dependencies to add/update\n"
            "3. Functions/classes to implement\n"
            "4. Test strategy\n"
            "5. Potential risks\n"
        )
        
        try:
            response = self.provider.complete([
                {"role": "system", "content": "You are a senior software architect. Be specific and detailed."},
                {"role": "user", "content": prompt},
            ])
            return response
        except Exception:
            return f"Plan: Implement the task '{task}' following existing patterns in the repository."
    
    def execute_tool_calls(self, llm_output: str) -> list[dict]:
        """Execute tool calls parsed from LLM output."""
        tool_calls = ToolCallParser.parse(llm_output)
        results = []
        
        for call in tool_calls:
            tool = self.tool_registry.get(call.get("name", ""))
            if not tool:
                results.append({"name": call.get("name"), "error": "Unknown tool"})
                continue
            
            # Validate args
            valid, error = self.tool_registry.validate_args(call["name"], call.get("params", {}))
            if not valid:
                results.append({"name": call["name"], "error": error})
                continue
            
            # Execute (placeholder — actual execution would use tool.handler)
            results.append({
                "name": call["name"],
                "params": call.get("params", {}),
                "result": f"[Executed {call['name']}]",
                "status": "ok",
            })
        
        return results
    
    def run_full_cycle(self, task: str, build_cmd: str = "", test_cmd: str = "") -> dict:
        """Run the complete Phase 6 cycle: Understand → Plan → Implement → QA → Review → Done."""
        t0 = time.time()
        cycle = {
            "task": task,
            "phases": {},
            "overall": {},
        }
        
        # Phase 1: Understand
        cycle["phases"]["understand"] = {
            "scan": self.scan_repository(),
            "architecture": self.analyze_architecture(),
        }
        
        # Phase 2: Plan
        cycle["phases"]["plan"] = self.plan_task(task)
        
        # Phase 3: Quality check (baseline)
        cycle["phases"]["quality_baseline"] = self.run_quality_pipeline()
        
        # Phase 4: Build & Test (if applicable)
        if build_cmd or test_cmd:
            cycle["phases"]["build_test"] = self.build_and_test(build_cmd, test_cmd)
        
        # Phase 5: Code Review
        cycle["phases"]["review"] = self.review_code()
        
        # Overall assessment
        scan = cycle["phases"]["understand"]["scan"]
        review = cycle["phases"]["review"]
        
        overall_status = "ready"
        if isinstance(review, dict) and review.get("blockers", 0) > 0:
            overall_status = "blockers_found"
        
        cycle["overall"] = {
            "status": overall_status,
            "files_analyzed": scan.get("files", 0),
            "symbols_indexed": scan.get("symbols", 0),
            "code_score": review.get("average_score", review.get("score", 5.0)),
            "duration_s": round(time.time() - t0, 1),
        }
        
        self.history.append(cycle)
        return cycle
    
    def generate_report(self) -> str:
        """Generate a comprehensive markdown report."""
        lines = []
        lines.append("# AI Software Engineer Report")
        lines.append(f"\n## Repository: {self.workspace}")
        lines.append(f"Files: {self.analyzer.metrics.get('total_files', 'N/A')}")
        lines.append(f"Symbols: {self.analyzer.metrics.get('total_symbols', 'N/A')}")
        
        if self.architect:
            arch = self.architect.analyze_current_architecture()
            lines.append(f"\n### Architecture")
            lines.append(f"Components: {len(arch.components)}")
            lines.append(f"Patterns: {', '.join(arch.patterns)}")
            
            if arch.risks:
                lines.append(f"\n### Risks ({len(arch.risks)})")
                for r in arch.risks:
                    lines.append(f"- ⚠ {r}")
            
            suggestions = self.architect.suggest_improvements()
            if suggestions:
                lines.append(f"\n### Improvement Suggestions")
                for s in suggestions:
                    lines.append(f"- [{s['type']}] {s['reason']}")
        
        if self.history:
            last = self.history[-1]
            lines.append(f"\n### Last Cycle")
            lines.append(f"Overall: {last['overall'].get('status', 'N/A')}")
            lines.append(f"Code Score: {last['overall'].get('code_score', 'N/A')}")
            lines.append(f"Duration: {last['overall'].get('duration_s', 'N/A')}s")
        
        return "\n".join(lines)
