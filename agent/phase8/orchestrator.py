"""
Phase 8: Orchestrator — Ties all self-improvement modules into a unified pipeline.

The orchestrator:
  ✅ Coordinates SelfEvaluator, ImprovementCycle, MemoryEvolution, OnlineLearning, MetaLearning
  ✅ Runs autonomous improvement cycles
  ✅ Manages the improvement → learning → meta-learning loop
  ✅ Generates improvement reports
  ✅ Integrates with the main agent system
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from .self_improve import SelfEvaluator, ImprovementCycle, ImprovementRecord, ImprovementStatus
from .memory_evolution import ExperienceReplay, FailureDatabase, KnowledgeCompressor, Experience
from .online_learning import OnlineLearner
from .meta_learning import MetaLearner


@dataclass
class SelfImprovementReport:
    """Comprehensive report of a self-improvement session."""
    session_id: str
    timestamp: float = field(default_factory=time.time)
    cycles_completed: int = 0
    initial_score: float = 0.0
    final_score: float = 0.0
    total_improvement: float = 0.0
    status: str = "unknown"
    new_skills_learned: int = 0
    failures_fixed: int = 0
    adapter_updated: bool = False
    summary: str = ""
    details: list[dict] = field(default_factory=list)


class Phase8Orchestrator:
    """
    Unified orchestrator for Phase 8: Self-Improving AI.

    This ties together:
      - self_improve: Evaluation and improvement cycles
      - memory_evolution: Experience replay and failure database
      - online_learning: LoRA adapter updates
      - meta_learning: Cross-task pattern learning
    """

    def __init__(
        self,
        debugger,          # Phase 6 DebuggerAgent
        critic,            # Phase 6 CriticAgent
        data_dir: str = "data/phase8",
        max_cycles: int = 10,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Core modules
        self.evaluator = SelfEvaluator()
        self.improvement_cycle = ImprovementCycle(
            evaluator=self.evaluator,
            debugger=debugger,
            critic=critic,
            max_cycles=max_cycles,
        )

        # Memory modules
        self.experience_replay = ExperienceReplay(
            store_path=str(self.data_dir / "experience_replay.jsonl"),
        )
        self.failure_db = FailureDatabase(
            db_path=str(self.data_dir / "failure_db.jsonl"),
        )
        self.knowledge_compressor = KnowledgeCompressor()

        # Learning modules
        self.online_learner = OnlineLearner(
            adapter_dir=str(self.data_dir / "adapters"),
        )
        self.meta_learner = MetaLearner(
            store_path=str(self.data_dir / "meta_knowledge.json"),
        )

        self.reports: list[SelfImprovementReport] = []

    def run_improvement_session(
        self,
        execute_fix: Callable[[str, dict], dict],
        test_command: str = "",
        review_target: Optional[str] = None,
        task_type: str = "general",
        on_cycle: Optional[Callable[[ImprovementRecord], None]] = None,
    ) -> SelfImprovementReport:
        """
        Run a full self-improvement session.

        Flow:
          1. Query past experience for relevant strategies
          2. Run improvement cycles (evaluate → diagnose → fix → measure)
          3. Record experiences (success & failure)
          4. Update meta-knowledge
          5. Optionally update LoRA adapter
          6. Generate report
        """
        session_id = f"session_{int(time.time())}"
        report = SelfImprovementReport(session_id=session_id)

        # 0. Warm-up: check if we have relevant past experience
        past_strategies = self.experience_replay.get_successful_strategies(task_type)
        past_failures = self.experience_replay.get_failure_lessons(task_type)
        meta_strategy = self.meta_learner.suggest_strategy(task_type)

        # 1. Run improvement cycles
        result = self.improvement_cycle.run_full(
            execute_fix=execute_fix,
            test_command=test_command,
            review_target=review_target,
            on_cycle=on_cycle,
        )

        report.cycles_completed = result["cycles"]
        report.initial_score = result["initial_score"]
        report.final_score = result["final_score"]
        report.total_improvement = result["total_improvement"]
        report.status = result["status"]

        # 2. Record experiences from improvement history
        for record in self.improvement_cycle.history:
            outcome = "success" if record.delta > 0.5 else ("failure" if record.delta < -0.5 else "partial")
            exp = Experience(
                id="",
                task_type=task_type,
                context=f"Improvement cycle {record.attempt}: {record.diagnosis[:200]}",
                outcome=outcome,
                score=max(0, min(100, 50 + record.delta * 5)),
                strategy=record.strategy,
                error_pattern=record.diagnosis if outcome == "failure" else "",
                fix_pattern="\n".join(record.applied_changes) if outcome == "success" else "",
                tags=[task_type, "phase8", outcome],
            )
            self.experience_replay.add(exp)

            # Record failures in failure DB
            if outcome == "failure" and record.diagnosis:
                self.failure_db.record_failure(
                    error_message=record.diagnosis,
                    fix="\n".join(record.applied_changes) if record.applied_changes else "",
                )
                report.failures_fixed += 1

        # 3. Update meta-knowledge
        if result["total_improvement"] > 0:
            self.meta_learner.record_task_result(
                task_type=task_type,
                strategy=meta_strategy if meta_strategy else "improvement_cycle",
                score=result["final_score"],
                complexity=max(1, 10 - result["total_improvement"] / 10),
            )
            # Register successful strategies as skills
            for record in self.improvement_cycle.history:
                if record.delta > 1.0:
                    self.meta_learner.register_skill(
                        pattern_type=f"{task_type}_improvement",
                        trigger_keywords=record.diagnosis.split()[:10] if record.diagnosis else ["improvement"],
                        strategy=record.strategy,
                        success=True,
                        context=record.diagnosis,
                    )
                    report.new_skills_learned += 1

        # 4. Convert history to details
        report.details = result.get("history", [])

        # 5. Generate summary
        report.summary = self._generate_summary(report, past_strategies)
        self.reports.append(report)

        return report

    def run_autonomous_loop(
        self,
        execute_fix: Callable[[str, dict], dict],
        test_command: str = "",
        review_target: Optional[str] = None,
        task_type: str = "general",
        max_sessions: int = 5,
        on_report: Optional[Callable[[SelfImprovementReport], None]] = None,
    ) -> list[SelfImprovementReport]:
        """
        Run multiple self-improvement sessions autonomously.

        Each session builds on the learnings from previous ones.
        """
        all_reports = []

        for session_i in range(max_sessions):
            # Before session: inject meta-knowledge into prompt/context
            few_shot = self.meta_learner.few_shot_prompt(
                task_type=task_type,
                task_description=f"Self-improvement session {session_i + 1}",
            )

            report = self.run_improvement_session(
                execute_fix=execute_fix,
                test_command=test_command,
                review_target=review_target,
                task_type=task_type,
            )
            all_reports.append(report)

            if on_report:
                on_report(report)

            # Check if converged
            if report.status == "converged":
                break

            # Between sessions: apply online learning if improved significantly
            if report.total_improvement > 5.0:
                self.online_learner.set_baseline(report.final_score)

        return all_reports

    def get_improvement_summary(self) -> dict:
        """Get overall improvement statistics."""
        if not self.reports:
            return {"status": "no_data"}

        total_delta = sum(r.total_improvement for r in self.reports)
        sessions = len(self.reports)
        converged = sum(1 for r in self.reports if r.status == "converged")

        return {
            "total_sessions": sessions,
            "converged_sessions": converged,
            "total_improvement": round(total_delta, 1),
            "avg_improvement_per_session": round(total_delta / sessions, 1) if sessions else 0,
            "best_final_score": max(r.final_score for r in self.reports),
            "total_skills_learned": sum(r.new_skills_learned for r in self.reports),
            "total_failures_fixed": sum(r.failures_fixed for r in self.reports),
            "latest_status": self.reports[-1].status if self.reports else "unknown",
        }

    def save_state(self):
        """Save orchestrator state for resumption."""
        state = {
            "evaluator_history": self.evaluator.metrics_history[-100:],
            "reports": [
                {
                    "session_id": r.session_id,
                    "cycles_completed": r.cycles_completed,
                    "initial_score": r.initial_score,
                    "final_score": r.final_score,
                    "total_improvement": r.total_improvement,
                    "status": r.status,
                    "new_skills_learned": r.new_skills_learned,
                    "failures_fixed": r.failures_fixed,
                    "timestamp": r.timestamp,
                    "details": r.details,
                }
                for r in self.reports
            ],
            "summary": self.get_improvement_summary(),
        }
        (self.data_dir / "orchestrator_state.json").write_text(
            json.dumps(state, indent=2, ensure_ascii=False),
        )

    def _generate_summary(
        self,
        report: SelfImprovementReport,
        past_strategies: list[str],
    ) -> str:
        """Generate a human-readable summary of the improvement session."""
        delta_str = f"+{report.total_improvement:.1f}" if report.total_improvement >= 0 else f"{report.total_improvement:.1f}"

        lines = [
            f"# Self-Improvement Session: {report.session_id}",
            f"Status: **{report.status}**",
            f"Score: {report.initial_score:.1f} → {report.final_score:.1f} (Δ {delta_str})",
            f"Cycles: {report.cycles_completed}",
            f"New skills: {report.new_skills_learned}",
            f"Failures resolved: {report.failures_fixed}",
        ]

        if past_strategies:
            lines.append(f"\nPast strategies available: {len(past_strategies)}")

        if report.details:
            lines.append("\n## Cycle Details")
            for d in report.details:
                emoji = "📈" if d.get("delta", 0) > 0 else "📉" if d.get("delta", 0) < 0 else "➡️"
                lines.append(
                    f"- {emoji} Cycle {d.get('attempt')}: "
                    f"{d.get('before')} → {d.get('after')} "
                    f"(Δ {d.get('delta', 0):+.1f}) [{d.get('status', '?')}]"
                )

        return "\n".join(lines)
