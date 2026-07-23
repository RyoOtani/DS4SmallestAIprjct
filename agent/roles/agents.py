"""
Agent Role Architecture: Clean separation of concerns.

Each agent role has a single responsibility and clear interface:

  Planner   → "What should we do?" (task decomposition, strategy)
  Architect → "How should we structure it?" (design patterns, architecture)
  Coder     → "Write the code" (implementation)
  Reviewer  → "Is this code good?" (code review, best practices)
  Tester    → "Does it work?" (test generation, regression detection)
  Debugger  → "Why did it fail?" (error analysis, root cause)
  Security  → "Is it safe?" (vulnerability scanning)

This replaces the monolithic AutonomousCodingAgent with composable roles.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
from pathlib import Path


# ── Shared Types ─────────────────────────────────────────────────────────────

@dataclass
class Task:
    """A task to be executed by the agent system."""
    id: str
    description: str
    context: str = ""           # relevant code context
    constraints: list[str] = field(default_factory=list)
    priority: str = "medium"    # low, medium, high, critical


@dataclass
class Plan:
    """A plan produced by the Planner."""
    task_id: str
    steps: list[str]
    reasoning: str = ""
    alternatives: list[str] = field(default_factory=list)
    estimated_complexity: str = "medium"


@dataclass
class CodeChange:
    """A code change produced by the Coder."""
    file: str
    patch: str                 # unified diff
    description: str
    tests_added: list[str] = field(default_factory=list)


@dataclass
class Review:
    """A code review produced by the Reviewer."""
    approved: bool
    score: float               # 0-10
    strengths: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)  # must-fix
    suggestions: list[str] = field(default_factory=list)


@dataclass
class TestReport:
    """Test results from the Tester."""
    passed: bool
    total: int = 0
    passed_count: int = 0
    failed_count: int = 0
    regressions: list[str] = field(default_factory=list)
    coverage_pct: float = 0.0
    output: str = ""


@dataclass
class Diagnosis:
    """A diagnosis from the Debugger."""
    root_cause: str
    confidence: float          # 0-1
    fix_suggestion: str = ""
    related_code: list[str] = field(default_factory=list)
    similar_issues: list[str] = field(default_factory=list)


# ── Agent Roles ──────────────────────────────────────────────────────────────

class PlannerAgent:
    """
    Plans WHAT to do. Pure reasoning — no code writing.

    Input: Task + repository context
    Output: Plan (steps + strategy)
    """

    def plan(self, task: Task, context: dict = None) -> Plan:
        """
        Decompose a task into actionable steps.

        Uses repository intelligence (call graph, dependencies)
        to make informed decisions about which files to touch.
        """
        # In real usage, this would call an LLM with repo context
        steps = self._decompose(task, context or {})
        return Plan(
            task_id=task.id,
            steps=steps,
            reasoning=f"Analyzed {task.description[:100]}",
            estimated_complexity=self._estimate_complexity(steps),
        )

    def _decompose(self, task: Task, context: dict) -> list[str]:
        """Decompose task into ordered steps."""
        desc = task.description.lower()

        # Pattern-based decomposition (in practice: LLM-powered)
        if "fix" in desc or "bug" in desc:
            return [
                "1. Reproduce the bug with a minimal test case",
                "2. Identify the root cause via debugging/tracing",
                "3. Implement the fix",
                "4. Add regression test to prevent recurrence",
                "5. Verify all existing tests still pass",
            ]
        elif "refactor" in desc:
            return [
                "1. Analyze current code structure and dependencies",
                "2. Identify extraction/simplification targets",
                "3. Apply refactoring incrementally (one change per commit)",
                "4. Run full test suite after each step",
                "5. Update documentation if API changed",
            ]
        elif "feature" in desc or "implement" in desc or "add" in desc:
            return [
                "1. Define the interface/API contract",
                "2. Write failing tests (TDD)",
                "3. Implement minimum viable version",
                "4. Iterate until tests pass",
                "5. Add documentation and examples",
            ]
        else:
            return [
                "1. Understand the requirement",
                "2. Explore relevant existing code",
                "3. Design the solution",
                "4. Implement with tests",
                "5. Review and refine",
            ]

    def _estimate_complexity(self, steps: list[str]) -> str:
        n = len(steps)
        if n <= 3:
            return "low"
        elif n <= 5:
            return "medium"
        return "high"


class ArchitectAgent:
    """
    Designs HOW to structure the solution. Architecture decisions.

    Input: Plan + codebase knowledge
    Output: Architecture recommendation (files to create/modify, patterns to use)
    """

    def design(self, plan: Plan, codebase: dict = None) -> dict:
        """Propose architecture for the plan."""
        return {
            "pattern": "incremental change",
            "files_to_modify": [],
            "files_to_create": [],
            "dependencies_to_add": [],
            "risks": [],
            "design_notes": f"Architecture for: {plan.task_id}",
        }


class CoderAgent:
    """
    Writes the code. Implementation only — no planning, no reviewing.

    Input: Plan + architecture + context
    Output: CodeChange (patch)
    """

    def implement(self, plan: Plan, architecture: dict = None, context: dict = None) -> list[CodeChange]:
        """Generate code changes implementing the plan."""
        return []

    def apply_patch(self, change: CodeChange, sandbox) -> bool:
        """Apply a code change in a sandboxed environment."""
        import subprocess
        target = Path(change.file)
        try:
            result = subprocess.run(
                ["patch", "-p1"],
                input=change.patch,
                capture_output=True,
                text=True,
                cwd=str(target.parent),
                timeout=30,
            )
            return result.returncode == 0
        except Exception:
            return False


class ReviewerAgent:
    """
    Reviews code quality. Catches issues before they reach production.

    Input: CodeChange + codebase standards
    Output: Review (approved/rejected + feedback)
    """

    def review(self, changes: list[CodeChange], standards: dict = None) -> Review:
        """Review a set of code changes."""
        issues = []
        strengths = []
        blockers = []

        for change in changes:
            patch = change.patch

            # Automated checks
            if "TODO" in patch or "FIXME" in patch:
                issues.append(f"Unresolved TODO/FIXME in {change.file}")
            if "print(" in patch and "test" not in change.file:
                issues.append(f"Debug print statement in {change.file}")
            if "pass" in patch and len(patch) < 50:
                issues.append(f"Empty implementation (pass) in {change.file}")

            if change.tests_added:
                strengths.append(f"Tests included for {change.file}")
            if "docstring" in patch.lower() or '"""' in patch:
                strengths.append(f"Documentation present in {change.file}")

        score = 10.0 - len(issues) * 1.5 - len(blockers) * 3.0
        score = max(0, min(10, score))

        return Review(
            approved=len(blockers) == 0 and score >= 5.0,
            score=round(score, 1),
            strengths=strengths,
            issues=issues,
            blockers=blockers,
            suggestions=[f"Consider addressing: {i}" for i in issues[:3]],
        )


class TestRunnerAgent:
    """
    Runs tests. Detects regressions, measures coverage.

    Input: CodeChange + test suite
    Output: TestReport
    """

    def run_tests(self, changes: list[CodeChange], sandbox=None) -> TestReport:
        """Run affected tests against changes."""
        from agent.core.regression import RegressionRunner
        runner = RegressionRunner()

        changed_files = [c.file for c in changes]
        report = runner.run_affected(changed_files, sandbox)

        return TestReport(
            passed=report.failed == 0,
            total=report.total,
            passed_count=report.passed,
            failed_count=report.failed,
            regressions=[
                r.test_name for r in report.results
                if r.new_failure
            ],
        )


class DebuggerAgent:
    """
    Diagnoses failures. Finds root causes.

    Input: TestReport + CodeChange + error output
    Output: Diagnosis
    """

    def diagnose(self, report: TestReport, changes: list[CodeChange], error_output: str = "") -> Diagnosis:
        """Diagnose the root cause of a test failure."""
        # Pattern-based diagnosis
        error_lower = error_output.lower()

        if "syntaxerror" in error_lower:
            return Diagnosis(
                root_cause="Syntax error in modified code",
                confidence=0.95,
                fix_suggestion="Check for missing colons, parentheses, or indentation",
                related_code=[c.file for c in changes],
            )

        if "attributeerror" in error_lower or "has no attribute" in error_lower:
            return Diagnosis(
                root_cause="Attribute/method access on wrong type",
                confidence=0.85,
                fix_suggestion="Verify the object type before attribute access",
                related_code=[c.file for c in changes],
            )

        if "importerror" in error_lower or "modulenotfound" in error_lower:
            return Diagnosis(
                root_cause="Missing or incorrect import",
                confidence=0.90,
                fix_suggestion="Check import paths and package structure",
                related_code=[c.file for c in changes],
            )

        if "typeerror" in error_lower:
            return Diagnosis(
                root_cause="Type mismatch in function call or assignment",
                confidence=0.80,
                fix_suggestion="Check argument types and return values",
                related_code=[c.file for c in changes],
            )

        return Diagnosis(
            root_cause="Unknown failure — requires deeper analysis",
            confidence=0.3,
            fix_suggestion="Review error output and traceback",
        )


class SecurityReviewerAgent:
    """
    Reviews code for security vulnerabilities.

    Input: CodeChange
    Output: Review (security-focused)
    """

    VULN_PATTERNS = [
        (r"os\.system\s*\(", "OS command injection risk"),
        (r"subprocess\.\w+\s*\(\s*['\"].*\$", "Shell injection risk"),
        (r"eval\s*\(", "Code injection via eval()"),
        (r"exec\s*\(", "Code injection via exec()"),
        (r"__import__\s*\(\s*['\"]os['\"]\s*\)", "Suspicious os import"),
        (r"pickle\.load", "Insecure deserialization"),
        (r"yaml\.load\s*\((?!”) (?!.*Loader)", "Insecure YAML loading (use SafeLoader)"),
        (r"assert\s+.*password", "Password in assert — stripped in optimized mode"),
        (r"hardcoded.*(?:password|secret|token|key|api)", "Hardcoded credentials"),
        (r"SELECT.*\+.*FROM|INSERT.*\+", "Potential SQL injection"),
    ]

    def review(self, changes: list[CodeChange]) -> Review:
        """Security-focused code review."""
        import re
        issues = []
        blockers = []

        for change in changes:
            for pattern, description in self.VULN_PATTERNS:
                if re.search(pattern, change.patch, re.IGNORECASE):
                    issues.append(f"🔴 {description} in {change.file}")
                    blockers.append(f"SECURITY: {description} in {change.file}")

        score = 10.0 - len(blockers) * 5.0 - len(issues) * 1.0
        score = max(0, min(10, score))

        return Review(
            approved=len(blockers) == 0,
            score=round(score, 1),
            issues=issues,
            blockers=blockers,
            suggestions=["Run a full SAST scan before deployment"] if issues else [],
        )
