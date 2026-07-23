"""
Tests for Phase 8: Self-Improving AI System.

Covers:
  - SelfEvaluator (scoring, trends)
  - ImprovementCycle (cycles, convergence)
  - ExperienceReplay (add, sample, strategies)
  - FailureDatabase (record, lookup, fixes)
  - KnowledgeCompressor (conversation, code changes, knowledge extraction)
  - OnlineLearner (snapshots, evaluation, rollback, guardrails, A/B testing)
  - MetaLearner (skill registration, matching, strategy suggestion, few-shot)
  - Phase8Orchestrator (sessions, reports, summary)
"""

from __future__ import annotations
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure we can import from agent.phase8
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.phase8.self_improve import (
    SelfEvaluator,
    ImprovementCycle,
    ImprovementRecord,
    ImprovementStatus,
)
from agent.phase8.memory_evolution import (
    Experience,
    ExperienceReplay,
    FailureDatabase,
    FailurePattern,
    KnowledgeCompressor,
)
from agent.phase8.online_learning import OnlineLearner, LoRASnapshot
from agent.phase8.meta_learning import MetaLearner, SkillTemplate, TaskProfile
from agent.phase8.orchestrator import Phase8Orchestrator, SelfImprovementReport


# ═══════════════════════════════════════════════════════════════
# Test Helpers
# ═══════════════════════════════════════════════════════════════

def make_mock_debugger():
    debugger = MagicMock()
    debugger.run_build.return_value = MagicMock(status="ok", diagnosis="")
    return debugger

def make_mock_critic():
    critic = MagicMock()
    mock_review = MagicMock()
    mock_review.score = 8.0
    mock_review.blockers = []
    critic.review_file.return_value = mock_review
    return critic


# ═══════════════════════════════════════════════════════════════
# SelfEvaluator Tests
# ═══════════════════════════════════════════════════════════════

class TestSelfEvaluator:
    def test_evaluate_empty(self):
        evaluator = SelfEvaluator()
        result = evaluator.evaluate(test_command="echo ok")
        assert "total_score" in result
        assert result["total_score"] == 70.0  # 40% tests pass(100) + 30% review(0, no review) + 20% quality(100) + 10% perf(100) = 70

    def test_evaluate_with_review(self):
        evaluator = SelfEvaluator()
        result = evaluator.evaluate(
            test_command="echo ok",
            review_result={"score": 90},
            quality_result={"lint": {"errors": 0, "warnings": 0}},
        )
        assert result["total_score"] > 70  # should be high

    def test_evaluate_with_lint_errors(self):
        evaluator = SelfEvaluator()
        result = evaluator.evaluate(
            test_command="echo ok",
            quality_result={"lint": {"errors": 5, "warnings": 0}},
        )
        assert result["components"]["quality"]["score"] == 50  # 100 - 5*10

    def test_trend_improving(self):
        evaluator = SelfEvaluator()
        # Simulate improving scores
        evaluator.metrics_history = [
            {"total_score": 50}, {"total_score": 60}, {"total_score": 70},
            {"total_score": 80}, {"total_score": 90},
        ]
        assert evaluator.get_trend() == "improving"

    def test_trend_stable(self):
        evaluator = SelfEvaluator()
        evaluator.metrics_history = [
            {"total_score": 75}, {"total_score": 76}, {"total_score": 74},
        ]
        assert evaluator.get_trend() == "stable"

    def test_trend_degrading(self):
        evaluator = SelfEvaluator()
        evaluator.metrics_history = [
            {"total_score": 90}, {"total_score": 80}, {"total_score": 70},
        ]
        assert evaluator.get_trend() == "degrading"

    def test_trend_insufficient(self):
        evaluator = SelfEvaluator()
        assert evaluator.get_trend() == "insufficient_data"


# ═══════════════════════════════════════════════════════════════
# ImprovementCycle Tests
# ═══════════════════════════════════════════════════════════════

class TestImprovementCycle:
    def test_run_cycle_basic(self):
        evaluator = SelfEvaluator()
        debugger = make_mock_debugger()
        critic = make_mock_critic()

        cycle = ImprovementCycle(evaluator, debugger, critic, max_cycles=5)

        def dummy_fix(diagnosis, context):
            return {"files_changed": ["test.py"]}

        record = cycle.run_cycle(
            execute_fix=dummy_fix,
            test_command="echo ok",
        )

        assert isinstance(record, ImprovementRecord)
        assert record.attempt == 1
        assert record.before_score is not None
        assert record.after_score is not None
        assert record.status in (
            ImprovementStatus.IMPROVED,
            ImprovementStatus.STAGNATED,
            ImprovementStatus.CONVERGED,
        )

    def test_run_full_converges(self):
        evaluator = SelfEvaluator()
        debugger = make_mock_debugger()
        critic = make_mock_critic()

        cycle = ImprovementCycle(
            evaluator, debugger, critic,
            max_cycles=5,
            convergence_cycles=2,
            improvement_threshold=99,  # essentially no improvement possible
        )

        def dummy_fix(diagnosis, context):
            return {"files_changed": []}

        result = cycle.run_full(
            execute_fix=dummy_fix,
            test_command="echo ok",
        )

        assert result["cycles"] <= 5
        assert "initial_score" in result
        assert "final_score" in result
        assert result["status"] in ("converged", "stagnated", "improved")

    def test_improvement_record_fields(self):
        record = ImprovementRecord(
            attempt=1,
            strategy="test_strategy",
            before_score=50,
            after_score=70,
            delta=20,
            status=ImprovementStatus.IMPROVED,
        )
        assert record.delta == 20
        assert record.status == ImprovementStatus.IMPROVED


# ═══════════════════════════════════════════════════════════════
# ExperienceReplay Tests
# ═══════════════════════════════════════════════════════════════

class TestExperienceReplay:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.tmpdir, "exp_replay.jsonl")

    def test_add_and_sample(self):
        replay = ExperienceReplay(capacity=100, store_path=self.store_path)
        exp = Experience(
            id="",
            task_type="code_generation",
            context="Test task",
            outcome="success",
            score=85,
            strategy="iterative_refinement",
        )
        replay.add(exp)

        samples = replay.sample(task_type="code_generation", n=5)
        assert len(samples) == 1
        assert samples[0].outcome == "success"
        assert samples[0].strategy == "iterative_refinement"

    def test_filter_by_outcome(self):
        replay = ExperienceReplay(capacity=100, store_path=self.store_path)
        for i in range(5):
            replay.add(Experience(
                id="", task_type="test", context=f"task_{i}",
                outcome="success" if i % 2 == 0 else "failure",
                score=50 + i * 10, strategy="s",
            ))

        successes = replay.sample(outcome="success", n=10)
        assert all(e.outcome == "success" for e in successes)
        assert len(successes) == 3

    def test_strategies(self):
        replay = ExperienceReplay(capacity=100, store_path=self.store_path)
        replay.add(Experience(
            id="", task_type="code_generation", context="a",
            outcome="success", score=90, strategy="strategy_A",
        ))
        replay.add(Experience(
            id="", task_type="code_generation", context="b",
            outcome="success", score=95, strategy="strategy_B",
        ))

        strategies = replay.get_successful_strategies("code_generation")
        assert "strategy_A" in strategies or "strategy_B" in strategies

    def test_failure_lessons(self):
        replay = ExperienceReplay(capacity=100, store_path=self.store_path)
        replay.add(Experience(
            id="", task_type="bug_fix", context="null pointer",
            outcome="failure", score=10, strategy="naive",
            error_pattern="NullPointerException at line 42",
        ))

        lessons = replay.get_failure_lessons("bug_fix")
        assert len(lessons) == 1
        assert "NullPointerException" in lessons[0]

    def test_capacity_eviction(self):
        replay = ExperienceReplay(capacity=3, store_path=self.store_path)
        for i in range(5):
            replay.add(Experience(
                id="", task_type="t", context=f"c_{i}",
                outcome="success", score=i * 10, strategy="s",
            ))
        assert len(replay.experiences) <= 3

    def test_replay_count_increments(self):
        replay = ExperienceReplay(capacity=100, store_path=self.store_path)
        replay.add(Experience(
            id="", task_type="t", context="c",
            outcome="success", score=50, strategy="s",
        ))
        assert replay.experiences[0].replay_count == 0
        replay.sample(n=1)
        assert replay.experiences[0].replay_count == 1


# ═══════════════════════════════════════════════════════════════
# FailureDatabase Tests
# ═══════════════════════════════════════════════════════════════

class TestFailureDatabase:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "failures.jsonl")

    def test_record_and_lookup(self):
        db = FailureDatabase(db_path=self.db_path)
        db.record_failure(
            error_message="SyntaxError: invalid syntax at line 10",
            file="test.py",
            line=10,
            fix="Add missing colon",
        )

        results = db.lookup("SyntaxError: invalid syntax at line 10")
        assert len(results) == 1
        assert results[0].occurrence_count == 1

    def test_duplicate_detection(self):
        db = FailureDatabase(db_path=self.db_path)
        for _ in range(3):
            db.record_failure(
                error_message="TypeError: NoneType has no attribute 'x'",
                fix="Add null check",
            )
        assert len(db.patterns) == 1
        assert db.patterns[list(db.patterns.keys())[0]].occurrence_count == 3

    def test_common_fixes(self):
        db = FailureDatabase(db_path=self.db_path)
        db.record_failure(
            error_message="ImportError: No module named 'torch'",
            fix="pip install torch",
        )
        fixes = db.get_common_fixes("ImportError: No module named 'torch'")
        assert "pip install torch" in fixes

    def test_fuzzy_match(self):
        db = FailureDatabase(db_path=self.db_path)
        db.record_failure(
            error_message="IndexError: list index out of range in process_data",
            fix="Check len before access",
        )
        results = db.lookup("IndexError: list index out of range in another function")
        assert len(results) >= 1  # fuzzy match on IndexError

    def test_load_save_persistence(self):
        db = FailureDatabase(db_path=self.db_path)
        db.record_failure(error_message="Error A", fix="Fix A")
        db.record_failure(error_message="Error B", fix="Fix B")

        # Reload
        db2 = FailureDatabase(db_path=self.db_path)
        assert len(db2.patterns) == 2


# ═══════════════════════════════════════════════════════════════
# KnowledgeCompressor Tests
# ═══════════════════════════════════════════════════════════════

class TestKnowledgeCompressor:
    def test_compress_conversation(self):
        kc = KnowledgeCompressor()
        messages = [
            {"role": "user", "content": "Fix the bug in auth.py"},
            {"role": "assistant", "content": "I found an error: missing import"},
            {"role": "assistant", "content": "Fixed and tested — all 12 tests pass"},
        ]
        result = kc.compress_conversation(messages)
        assert "error" in result
        assert "fixed" in result.lower()
        assert "auth.py" in result

    def test_compress_empty(self):
        kc = KnowledgeCompressor()
        assert kc.compress_conversation([]) == ""

    def test_compress_code_changes(self):
        kc = KnowledgeCompressor()
        changes = [
            {"file": "src/main.py", "action": "modified", "summary": "Fixed off by one error"},
            {"file": "tests/test_main.py", "action": "created", "summary": "Added unit tests"},
        ]
        result = kc.compress_code_changes(changes)
        assert "src/main.py" in result
        assert "modified" in result
        assert "tests/test_main.py" in result
        assert "created" in result

    def test_extract_knowledge(self):
        kc = KnowledgeCompressor()
        experiences = [
            Experience(id="1", task_type="code_gen", context="a", outcome="success",
                       score=90, strategy="iterative", error_pattern=""),
            Experience(id="2", task_type="code_gen", context="b", outcome="failure",
                       score=20, strategy="one_shot", error_pattern="Null pointer"),
        ]
        result = kc.extract_knowledge(experiences)
        assert "Successful" in result
        assert "failures" in result.lower()
        assert "iterative" in result

    def test_extract_knowledge_empty(self):
        kc = KnowledgeCompressor()
        assert kc.extract_knowledge([]) == "No prior knowledge."


# ═══════════════════════════════════════════════════════════════
# OnlineLearner Tests
# ═══════════════════════════════════════════════════════════════

class TestOnlineLearner:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_snapshot_management(self):
        learner = OnlineLearner(adapter_dir=self.tmpdir, max_snapshots=5)
        learner.set_baseline(50)

        snap = learner.create_snapshot({"lora_A": [1, 2, 3]}, score=55, description="V1")
        assert snap.version == 1
        assert learner.current_version == 1
        assert learner.get_latest().score == 55

    def test_evaluate_update_keep(self):
        learner = OnlineLearner(adapter_dir=self.tmpdir)
        keep, reason = learner.evaluate_update(before_score=50, after_score=55)
        assert keep is True
        assert "Improved" in reason

    def test_evaluate_update_rollback(self):
        learner = OnlineLearner(adapter_dir=self.tmpdir)
        keep, reason = learner.evaluate_update(before_score=50, after_score=40)
        assert keep is False
        assert "ROLLBACK" in reason

    def test_evaluate_update_neutral(self):
        learner = OnlineLearner(adapter_dir=self.tmpdir)
        keep, reason = learner.evaluate_update(before_score=50, after_score=50.5)
        assert keep is True
        assert "Neutral" in reason

    def test_rollback(self):
        learner = OnlineLearner(adapter_dir=self.tmpdir, max_snapshots=10)
        learner.create_snapshot({"v": 1}, score=60, description="Good")
        learner.create_snapshot({"v": 2}, score=45, description="Bad")

        good = learner.rollback()
        assert good is not None
        assert good.score == 60
        assert len(learner.snapshots) == 1

    def test_get_best(self):
        learner = OnlineLearner(adapter_dir=self.tmpdir, max_snapshots=10)
        learner.create_snapshot({"v": 1}, score=50)
        learner.create_snapshot({"v": 2}, score=90)
        learner.create_snapshot({"v": 3}, score=70)

        best = learner.get_best()
        assert best.score == 90

    def test_ab_test(self):
        learner = OnlineLearner()

        def eval_fn(state):
            return {"a": 50, "b": 80}.get(state.get("label", ""), 0)

        result = learner.a_b_test(
            variant_a={"label": "a"},
            variant_b={"label": "b"},
            eval_fn=eval_fn,
        )
        assert result["winner"] == "B"
        assert result["score_b"] > result["score_a"]

    def test_guardrails_harmful(self):
        learner = OnlineLearner()
        data = [
            {"text": "normal code", "score": 5},
            {"text": "rm -rf / dangerous", "score": 5},
            {"text": "eval(base64 encoded malware)", "score": 3},
            {"text": "DROP TABLE users;", "score": 2},
        ]
        safe = learner.apply_guardrails(data)
        assert len(safe) == 1
        assert safe[0]["text"] == "normal code"

    def test_guardrails_low_quality(self):
        learner = OnlineLearner()
        data = [
            {"text": "good", "score": 8},
            {"text": "bad", "score": -5},  # rejected
            {"text": "ok", "score": 0},
        ]
        safe = learner.apply_guardrails(data)
        assert len(safe) == 2

    def test_guardrails_dedup(self):
        learner = OnlineLearner()
        data = [
            {"text": "same content", "score": 5},
            {"text": "same content", "score": 5},
        ]
        safe = learner.apply_guardrails(data)
        assert len(safe) == 1

    def test_max_rollbacks(self):
        learner = OnlineLearner(adapter_dir=self.tmpdir, max_rollbacks=2)
        # Simulate 3 rollbacks
        learner._rollback_count = 0
        keep1, _ = learner.evaluate_update(50, 40)
        keep2, _ = learner.evaluate_update(50, 40)
        keep3, _ = learner.evaluate_update(50, 40)
        # Third should mention max reached
        assert "max reached" in _


# ═══════════════════════════════════════════════════════════════
# MetaLearner Tests
# ═══════════════════════════════════════════════════════════════

class TestMetaLearner:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.tmpdir, "meta.json")

    def test_register_and_match_skill(self):
        meta = MetaLearner(store_path=self.store_path)
        meta.register_skill(
            pattern_type="bug_pattern",
            trigger_keywords=["null", "pointer", "None", "check"],
            strategy="add_null_check",
            success=True,
            context="Fixed null pointer bug",
            code="if x is None: return",
        )

        skills = meta.match_skills("We have a null pointer issue when accessing None")
        assert len(skills) == 1
        assert skills[0].strategy == "add_null_check"

    def test_match_min_rate_filter(self):
        meta = MetaLearner(store_path=self.store_path)
        meta.register_skill(
            pattern_type="bug", trigger_keywords=["null"],
            strategy="check", success=False,
        )
        # Low success rate should be filtered
        skills = meta.match_skills("null issue", min_success_rate=0.8)
        assert len(skills) == 0

    def test_suggest_strategy(self):
        meta = MetaLearner(store_path=self.store_path)
        # Build up a task profile
        for _ in range(15):
            meta.record_task_result(
                task_type="code_generation",
                strategy="iterative_refinement",
                score=85,
            )
        for _ in range(5):
            meta.record_task_result(
                task_type="code_generation",
                strategy="one_shot",
                score=45,
            )

        strategy = meta.suggest_strategy("code_generation")
        assert strategy == "iterative_refinement"

    def test_strategy_ranking(self):
        meta = MetaLearner(store_path=self.store_path)
        meta.record_task_result("bug_fix", "strategy_a", 90)
        meta.record_task_result("bug_fix", "strategy_b", 70)

        ranking = meta.get_strategy_ranking("bug_fix")
        assert ranking[0][0] == "strategy_a"
        assert ranking[0][1] > ranking[1][1]

    def test_few_shot_prompt(self):
        meta = MetaLearner(store_path=self.store_path)
        meta.register_skill(
            pattern_type="refactor",
            trigger_keywords=["refactor", "clean", "restructure"],
            strategy="extract_method_then_inline",
            success=True,
            context="Clean up main.py",
            code="def clean(): pass",
        )

        prompt = meta.few_shot_prompt(
            task_type="refactoring",
            task_description="Refactor the utils module",
        )
        assert "Refactor the utils module" in prompt
        assert "extract_method_then_inline" in prompt

    def test_meta_optimize_prompt(self):
        meta = MetaLearner(store_path=self.store_path)
        meta.record_task_result("code_gen", "top_down", 90)
        meta.record_task_result("code_gen", "bottom_up", 70)

        def eval_fn(prompt):
            # Score higher if "top_down" is mentioned
            return 80 if "top_down" in prompt.lower() else 50

        best_prompt, best_score = meta.meta_optimize_prompt(
            task_type="code_gen",
            current_prompt="Generate code for the task.",
            evaluate_fn=eval_fn,
            iterations=3,
        )
        assert best_score >= 80
        assert "top_down" in best_prompt.lower()

    def test_persistence(self):
        meta = MetaLearner(store_path=self.store_path)
        meta.register_skill(
            pattern_type="test", trigger_keywords=["test"], strategy="test",
            success=True,
        )
        meta.record_task_result("test_type", "strat", 80)

        meta2 = MetaLearner(store_path=self.store_path)
        assert len(meta2.skills) == 1
        assert len(meta2.task_profiles) == 1


# ═══════════════════════════════════════════════════════════════
# Phase8Orchestrator Tests
# ═══════════════════════════════════════════════════════════════

class TestPhase8Orchestrator:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmpdir, "phase8_data")
        self.debugger = make_mock_debugger()
        self.critic = make_mock_critic()

    def test_init(self):
        orch = Phase8Orchestrator(
            debugger=self.debugger,
            critic=self.critic,
            data_dir=self.data_dir,
        )
        assert orch.evaluator is not None
        assert orch.experience_replay is not None
        assert orch.failure_db is not None
        assert orch.online_learner is not None
        assert orch.meta_learner is not None

    def test_run_improvement_session(self):
        orch = Phase8Orchestrator(
            debugger=self.debugger,
            critic=self.critic,
            data_dir=self.data_dir,
            max_cycles=3,
        )

        def dummy_fix(diagnosis, context):
            return {"files_changed": ["test.py"]}

        report = orch.run_improvement_session(
            execute_fix=dummy_fix,
            test_command="echo ok",
            task_type="bug_fix",
        )

        assert isinstance(report, SelfImprovementReport)
        assert report.cycles_completed > 0
        assert report.status in ("converged", "stagnated", "improved")

    def test_get_improvement_summary_no_data(self):
        orch = Phase8Orchestrator(
            debugger=self.debugger,
            critic=self.critic,
            data_dir=self.data_dir,
        )
        summary = orch.get_improvement_summary()
        assert summary["status"] == "no_data"

    def test_get_improvement_summary_with_data(self):
        orch = Phase8Orchestrator(
            debugger=self.debugger,
            critic=self.critic,
            data_dir=self.data_dir,
            max_cycles=2,
        )

        def dummy_fix(d, c):
            return {"files_changed": []}

        orch.run_improvement_session(
            execute_fix=dummy_fix,
            test_command="echo ok",
            task_type="test",
        )

        summary = orch.get_improvement_summary()
        assert summary["total_sessions"] == 1
        assert summary["latest_status"] != "no_data"

    def test_save_state(self):
        orch = Phase8Orchestrator(
            debugger=self.debugger,
            critic=self.critic,
            data_dir=self.data_dir,
            max_cycles=1,
        )

        def dummy_fix(d, c):
            return {"files_changed": []}

        orch.run_improvement_session(
            execute_fix=dummy_fix,
            test_command="echo ok",
            task_type="test",
        )
        orch.save_state()

        state_path = Path(self.data_dir) / "orchestrator_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        assert "summary" in state

    def test_report_summary_string(self):
        orch = Phase8Orchestrator(
            debugger=self.debugger,
            critic=self.critic,
            data_dir=self.data_dir,
            max_cycles=1,
        )

        def dummy_fix(d, c):
            return {"files_changed": []}

        report = orch.run_improvement_session(
            execute_fix=dummy_fix,
            test_command="echo ok",
            task_type="test",
        )
        assert report.summary
        assert "Self-Improvement" in report.summary
        assert report.status in report.summary


# ═══════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
