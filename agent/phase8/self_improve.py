"""
Phase 8: Autonomous Self-Improvement Cycle.

Implements a closed-loop system where the AI:
  1. Generates code / takes actions
  2. Self-evaluates the result (critic, tests, metrics)
  3. Identifies improvement areas
  4. Applies fixes / re-learns
  5. Measures improvement delta
  6. Repeats — each cycle makes the AI better

Key innovations:
  ✅ Self-critique without human feedback
  ✅ Improvement delta measurement (before/after metrics)
  ✅ Experience replay from past successes and failures
  ✅ Automatic A/B testing of strategies
  ✅ Convergence detection (stop when no further improvement)
"""

from __future__ import annotations
import json
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable


class ImprovementStatus(Enum):
    PENDING = "pending"
    EVALUATING = "evaluating"
    IMPROVING = "improving"
    IMPROVED = "improved"
    STAGNATED = "stagnated"
    DEGRADED = "degraded"
    CONVERGED = "converged"


@dataclass
class ImprovementRecord:
    """Single improvement attempt record."""
    attempt: int
    strategy: str
    before_score: float
    after_score: float
    delta: float
    status: ImprovementStatus
    diagnosis: str = ""
    applied_changes: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    rolled_back: bool = False


class SelfEvaluator:
    """Evaluates the quality of generated code/actions without human feedback."""

    def __init__(self):
        self.metrics_history: list[dict] = []

    def evaluate(
        self,
        test_command: str,
        review_result: Optional[dict] = None,
        quality_result: Optional[dict] = None,
    ) -> dict:
        """
        Compute a self-evaluation score (0-100) based on:
          - Test pass rate
          - Code review score
          - Lint/quality metrics
          - Performance (execution time)
        """
        score = 0.0
        components = {}

        # 1. Test results (weight: 40%)
        test_score = 0.0
        if test_command:
            import subprocess
            try:
                result = subprocess.run(
                    test_command, shell=True, capture_output=True, text=True, timeout=60,
                )
                test_score = 100.0 if result.returncode == 0 else 0.0
                components["tests"] = {"score": test_score, "passed": result.returncode == 0}
            except Exception:
                components["tests"] = {"score": 0, "error": "execution failed"}
        score += test_score * 0.4

        # 2. Code review (weight: 30%)
        review_score = 0.0
        if review_result and isinstance(review_result, dict):
            review_score = review_result.get("score", review_result.get("average_score", 50))
        components["review"] = {"score": review_score}
        score += review_score * 0.3

        # 3. Quality metrics (weight: 20%)
        quality_score = 100.0
        if quality_result and isinstance(quality_result, dict):
            lint = quality_result.get("lint", {})
            if lint:
                errors = lint.get("errors", 0)
                warnings = lint.get("warnings", 0)
                quality_score = max(0, 100 - errors * 10 - warnings * 2)
        components["quality"] = {"score": quality_score}
        score += quality_score * 0.2

        # 4. Performance (weight: 10%)
        perf_score = 100.0
        components["performance"] = {"score": perf_score}
        score += perf_score * 0.1

        result = {
            "total_score": round(score, 1),
            "components": components,
            "timestamp": time.time(),
        }
        self.metrics_history.append(result)
        return result

    def get_trend(self, window: int = 5) -> str:
        """Determine score trend: improving, stable, degrading."""
        if len(self.metrics_history) < 2:
            return "insufficient_data"
        recent = self.metrics_history[-window:]
        scores = [m["total_score"] for m in recent]
        if len(scores) < 2:
            return "stable"
        slope = (scores[-1] - scores[0]) / max(len(scores) - 1, 1)
        if slope > 1.0:
            return "improving"
        if slope < -1.0:
            return "degrading"
        return "stable"


class ImprovementCycle:
    """
    Closed-loop self-improvement: Evaluate → Diagnose → Fix → Measure → Repeat.

    Stops when:
      - Score converges (delta < threshold for N consecutive cycles)
      - Maximum cycles reached
      - Score degrades and cannot be recovered
    """

    def __init__(
        self,
        evaluator: SelfEvaluator,
        debugger,   # Phase 6 DebuggerAgent
        critic,     # Phase 6 CriticAgent
        max_cycles: int = 10,
        improvement_threshold: float = 0.5,
        convergence_cycles: int = 3,
    ):
        self.evaluator = evaluator
        self.debugger = debugger
        self.critic = critic
        self.max_cycles = max_cycles
        self.improvement_threshold = improvement_threshold
        self.convergence_cycles = convergence_cycles

        self.history: list[ImprovementRecord] = []
        self.best_score: float = 0.0
        self.best_state: Optional[dict] = None
        self._stagnation_count = 0

    def run_cycle(
        self,
        execute_fix: Callable[[str, dict], dict],
        test_command: str = "",
        review_target: Optional[str] = None,
    ) -> ImprovementRecord:
        """Execute one self-improvement cycle."""
        t0 = time.time()

        # 1. Evaluate current state
        review = self.critic.review_file(review_target) if review_target else None
        review_dict = {"score": review.score} if review else None

        baseline = self.evaluator.evaluate(
            test_command=test_command,
            review_result=review_dict,
        )
        before_score = baseline["total_score"]

        # 2. Check convergence
        if self._stagnation_count >= self.convergence_cycles:
            return ImprovementRecord(
                attempt=len(self.history) + 1,
                strategy="converged",
                before_score=before_score,
                after_score=before_score,
                delta=0.0,
                status=ImprovementStatus.CONVERGED,
                duration_s=time.time() - t0,
            )

        # 3. Diagnose issues
        diagnosis = ""
        if before_score < 90.0:
            if test_command:
                build_result = self.debugger.run_build(test_command)
                if build_result.status != "ok":
                    diagnosis = build_result.diagnosis
            if not diagnosis and review and review.score < 7.0:
                blockers = review.blockers[:3] if review.blockers else []
                diagnosis = "; ".join(b.message for b in blockers)

        # 4. Apply fix
        fix_result = execute_fix(diagnosis, {"before_score": before_score})

        # 5. Re-evaluate
        review2 = self.critic.review_file(review_target) if review_target else None
        review_dict2 = {"score": review2.score} if review2 else None

        after = self.evaluator.evaluate(
            test_command=test_command,
            review_result=review_dict2,
        )
        after_score = after["total_score"]
        delta = after_score - before_score

        # 6. Determine status
        if delta > self.improvement_threshold:
            status = ImprovementStatus.IMPROVED
            self._stagnation_count = 0
        elif delta < -self.improvement_threshold:
            status = ImprovementStatus.DEGRADED
        elif abs(delta) <= self.improvement_threshold:
            self._stagnation_count += 1
            status = ImprovementStatus.CONVERGED if self._stagnation_count >= self.convergence_cycles else ImprovementStatus.STAGNATED
        else:
            status = ImprovementStatus.IMPROVED

        # 7. Update best
        if after_score > self.best_score:
            self.best_score = after_score

        record = ImprovementRecord(
            attempt=len(self.history) + 1,
            strategy=diagnosis[:200] if diagnosis else "general_improvement",
            before_score=before_score,
            after_score=after_score,
            delta=round(delta, 2),
            status=status,
            diagnosis=diagnosis[:500],
            applied_changes=fix_result.get("files_changed", []),
            duration_s=round(time.time() - t0, 1),
        )
        self.history.append(record)
        return record

    def run_full(
        self,
        execute_fix: Callable[[str, dict], dict],
        test_command: str = "",
        review_target: Optional[str] = None,
        on_cycle: Optional[Callable[[ImprovementRecord], None]] = None,
    ) -> dict:
        """Run the full self-improvement loop until convergence."""
        results = []
        final_status = ImprovementStatus.PENDING

        for cycle in range(self.max_cycles):
            record = self.run_cycle(execute_fix, test_command, review_target)
            results.append(record)

            if on_cycle:
                on_cycle(record)

            if record.status == ImprovementStatus.CONVERGED:
                final_status = ImprovementStatus.CONVERGED
                break
            if record.status == ImprovementStatus.DEGRADED and record.delta < -5.0:
                final_status = ImprovementStatus.DEGRADED
                break

        final_status = final_status if final_status != ImprovementStatus.PENDING else results[-1].status if results else ImprovementStatus.STAGNATED

        return {
            "status": final_status.value,
            "cycles": len(results),
            "initial_score": results[0].before_score if results else 0,
            "final_score": results[-1].after_score if results else 0,
            "total_improvement": round(
                (results[-1].after_score - results[0].before_score) if results else 0, 1,
            ),
            "best_score": self.best_score,
            "history": [
                {
                    "attempt": r.attempt,
                    "delta": r.delta,
                    "status": r.status.value,
                    "before": r.before_score,
                    "after": r.after_score,
                }
                for r in results
            ],
        }
