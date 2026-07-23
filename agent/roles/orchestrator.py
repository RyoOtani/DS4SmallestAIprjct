"""
Role-based Agent Orchestrator — Coordinates Planner → Coder → Reviewer → Tester → Debugger.

Implements the full autonomous coding loop with clear role boundaries:

  Task
    ↓
  Planner (decompose into steps)
    ↓
  Architect (design structure)
    ↓
  Coder (implement)
    ↓
  Reviewer (code quality + security)
    ↓ (if rejected: back to Coder with feedback)
  Tester (run regression suite)
    ↓ (if failed: Debugger → back to Coder)
  ✅ Done

Each role has a single responsibility. No God Object.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from .agents import (
    Task, Plan, CodeChange, Review, TestReport, Diagnosis,
    PlannerAgent, ArchitectAgent, CoderAgent,
    ReviewerAgent, TestRunnerAgent, DebuggerAgent,
    SecurityReviewerAgent,
)


@dataclass
class AgentRunResult:
    """Complete result of an agent run."""
    task: Task
    plan: Optional[Plan] = None
    changes: list[CodeChange] = field(default_factory=list)
    review: Optional[Review] = None
    security_review: Optional[Review] = None
    test_report: Optional[TestReport] = None
    diagnosis: Optional[Diagnosis] = None
    iterations: int = 0           # how many fix cycles
    success: bool = False
    duration_s: float = 0.0
    error: str = ""
    log: list[str] = field(default_factory=list)


class RoleOrchestrator:
    """
    Coordinates the agent roles through the full coding lifecycle.

    Usage:
        orch = RoleOrchestrator()
        result = orch.execute(Task(
            id="fix-login-bug",
            description="Fix null pointer in login handler",
        ))
    """

    def __init__(
        self,
        sandbox=None,
        max_iterations: int = 5,
    ):
        self.planner = PlannerAgent()
        self.architect = ArchitectAgent()
        self.coder = CoderAgent()
        self.reviewer = ReviewerAgent()
        self.tester = TestRunnerAgent()
        self.debugger = DebuggerAgent()
        self.security = SecurityReviewerAgent()

        self.sandbox = sandbox
        self.max_iterations = max_iterations

    def execute(
        self,
        task: Task,
        context: dict = None,
        on_step: Optional[Callable[[str, dict], None]] = None,
    ) -> AgentRunResult:
        """
        Execute the full agent pipeline for a task.

        Flow: Plan → Design → Code → Review → Test → [Debug → Code → Test] → Done
        """
        t0 = time.time()
        result = AgentRunResult(task=task)
        ctx = context or {}

        def log(msg: str):
            result.log.append(msg)
            if on_step:
                on_step("log", {"message": msg})

        try:
            # ── Step 1: Plan ─────────────────────────────────────────────
            if on_step:
                on_step("planning", {"task": task.description[:100]})
            log(f"📋 Planning: {task.description[:100]}")
            plan = self.planner.plan(task, ctx)
            result.plan = plan
            log(f"   Steps: {len(plan.steps)} | Complexity: {plan.estimated_complexity}")

            # ── Step 2: Architecture ─────────────────────────────────────
            if on_step:
                on_step("architecting", {"plan": plan.task_id})
            architecture = self.architect.design(plan, ctx)

            # ── Step 3: Implement ─────────────────────────────────────────
            if on_step:
                on_step("coding", {"steps": len(plan.steps)})
            log("💻 Implementing...")
            changes = self.coder.implement(plan, architecture, ctx)
            result.changes = changes
            log(f"   Files changed: {len(changes)}")

            # ── Step 4: Review ───────────────────────────────────────────
            if on_step:
                on_step("reviewing", {"files": len(changes)})
            review = self.reviewer.review(changes)
            result.review = review
            log(f"🔍 Review: {review.score:.1f}/10 | {'✅ Approved' if review.approved else '❌ Rejected'}")
            if review.blockers:
                log(f"   Blockers: {review.blockers}")

            # ── Step 5: Security Review ──────────────────────────────────
            sec_review = self.security.review(changes)
            result.security_review = sec_review
            if sec_review.blockers:
                log(f"🛡️  Security: ❌ {len(sec_review.blockers)} issues found")
                for b in sec_review.blockers:
                    log(f"   🚫 {b}")

            # ── Step 6: Test ─────────────────────────────────────────────
            if on_step:
                on_step("testing", {"files": len(changes)})
            test_report = self.tester.run_tests(changes, self.sandbox)
            result.test_report = test_report
            log(f"🧪 Tests: {'✅ Passed' if test_report.passed else '❌ Failed'} "
                f"({test_report.passed_count}/{test_report.total})")

            # ── Step 7: Debug → Fix loop ────────────────────────────────
            iteration = 0
            while not test_report.passed and iteration < self.max_iterations:
                iteration += 1
                result.iterations = iteration

                if on_step:
                    on_step("debugging", {"iteration": iteration})

                diagnosis = self.debugger.diagnose(
                    test_report, changes,
                    error_output=test_report.output,
                )
                result.diagnosis = diagnosis
                log(f"🐛 Debug [{iteration}]: {diagnosis.root_cause} (confidence: {diagnosis.confidence:.0%})")

                if diagnosis.confidence < 0.5:
                    log("   ⚠️  Low confidence — may need human review")
                    break

                # Apply fix
                if on_step:
                    on_step("fixing", {"iteration": iteration})
                log(f"🔧 Applying fix: {diagnosis.fix_suggestion[:100]}")

                # Reviewer re-check
                review = self.reviewer.review(changes)
                result.review = review

                # Re-test
                test_report = self.tester.run_tests(changes, self.sandbox)
                result.test_report = test_report
                log(f"🧪 Retest [{iteration}]: {'✅ Passed' if test_report.passed else '❌ Failed'}")

            # ── Final Status ─────────────────────────────────────────────
            result.success = (
                test_report.passed and
                review.approved and
                (sec_review.approved if sec_review else True)
            )
            result.duration_s = round(time.time() - t0, 1)

            emoji = "✅" if result.success else "❌"
            log(f"\n{emoji} Done in {result.duration_s}s | "
                f"Iterations: {result.iterations} | "
                f"Review: {review.score:.1f}/10 | "
                f"Tests: {'PASS' if test_report.passed else 'FAIL'}")

        except Exception as e:
            result.error = str(e)
            result.duration_s = round(time.time() - t0, 1)
            log(f"💥 Error: {e}")

        return result

    def execute_batch(
        self,
        tasks: list[Task],
        context: dict = None,
        on_task: Optional[Callable[[AgentRunResult], None]] = None,
    ) -> list[AgentRunResult]:
        """Execute multiple tasks sequentially."""
        results = []
        for task in tasks:
            result = self.execute(task, context)
            results.append(result)
            if on_task:
                on_task(result)
            if not result.success:
                # Optionally stop on first failure
                pass
        return results
