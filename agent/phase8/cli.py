#!/usr/bin/env python3
"""
Phase 8 CLI: Command-line interface for self-improvement operations.

Usage:
  python -m agent.phase8.cli improve    Run self-improvement cycle
  python -m agent.phase8.cli evaluate   Evaluate current code quality
  python -m agent.phase8.cli replay     Show experience replay insights
  python -m agent.phase8.cli failures   Show failure database
  python -m agent.phase8.cli skills     Show learned meta-skills
  python -m agent.phase8.cli adapters   Manage LoRA adapters
  python -m agent.phase8.cli status     Show overall improvement status
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.phase8 import (
    SelfEvaluator,
    ExperienceReplay,
    FailureDatabase,
    OnlineLearner,
    MetaLearner,
    Phase8Orchestrator,
)


def cmd_evaluate(args):
    """Evaluate current code quality."""
    evaluator = SelfEvaluator()
    test_cmd = args.test_command or "echo 'no tests'"
    result = evaluator.evaluate(test_command=test_cmd)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def cmd_replay(args):
    """Show experience replay insights."""
    replay = ExperienceReplay()
    task_type = args.task_type or None
    outcome = args.outcome or None

    experiences = replay.sample(
        task_type=task_type,
        outcome=outcome,
        n=args.n,
        prioritize_recent=not args.no_recent,
        prioritize_high_score=args.best,
    )

    print(f"\n=== Experience Replay ({len(experiences)} samples) ===\n")
    for exp in experiences:
        emoji = "✅" if exp.outcome == "success" else "❌" if exp.outcome == "failure" else "🔄"
        print(f"{emoji} [{exp.task_type}] {exp.context[:120]}")
        print(f"   Strategy: {exp.strategy[:100]}")
        print(f"   Score: {exp.score} | Replays: {exp.replay_count}")
        print()

    # Show successful strategies
    if task_type:
        strategies = replay.get_successful_strategies(task_type)
        if strategies:
            print(f"\n--- Top strategies for '{task_type}' ---")
            for s in strategies:
                print(f"  • {s}")


def cmd_failures(args):
    """Show failure database."""
    db = FailureDatabase()
    patterns = list(db.patterns.values())
    patterns.sort(key=lambda p: -p.occurrence_count)

    print(f"\n=== Failure Database ({len(patterns)} patterns) ===\n")
    for pat in patterns[:args.n]:
        print(f"🔴 [{pat.occurrence_count}x] {pat.description[:120]}")
        if pat.common_fixes:
            for fix in pat.common_fixes[:2]:
                print(f"   Fix: {fix[:100]}")
        if pat.affected_files:
            print(f"   Files: {', '.join(pat.affected_files[:3])}")
        print()


def cmd_skills(args):
    """Show learned meta-skills."""
    meta = MetaLearner()

    print(f"\n=== Learned Skills ({len(meta.skills)} total) ===\n")
    skills = sorted(meta.skills.values(), key=lambda s: -s.success_rate)
    for skill in skills[:args.n]:
        rate_bar = "█" * int(skill.success_rate * 10)
        print(f"📚 [{skill.pattern_type}] {skill.description[:100]}")
        print(f"   Success rate: {skill.success_rate:.0%} {rate_bar}")
        print(f"   Strategy: {skill.strategy[:120]}")
        print(f"   Used: {skill.use_count}x")
        print()

    # Task profiles
    print("--- Task Strategy Rankings ---")
    for task_type, profile in meta.task_profiles.items():
        ranking = meta.get_strategy_ranking(task_type)
        print(f"\n  {task_type} ({profile.sample_count} samples):")
        for strategy, score in ranking[:3]:
            print(f"    • {strategy}: {score:.1f}")


def cmd_adapters(args):
    """Manage LoRA adapters."""
    learner = OnlineLearner()

    latest = learner.get_latest()
    best = learner.get_best()

    print(f"\n=== LoRA Adapters ===\n")
    print(f"Adapter dir: {learner.adapter_dir}")
    print(f"Current version: {learner.current_version}")
    print(f"Baseline score: {learner.baseline_score:.1f}")
    print(f"Snapshots: {len(learner.snapshots)}")
    print()

    if latest:
        print(f"Latest: v{latest.version} (score: {latest.score:.1f})")
    if best:
        print(f"Best:   v{best.version} (score: {best.score:.1f})")
    if latest and best:
        print(f"Gap:    {best.score - latest.score:+.1f}")

    print("\n--- Snapshots ---")
    for snap in learner.snapshots[-10:]:
        marker = " ← BEST" if best and snap.version == best.version else ""
        marker = " ← LATEST" if latest and snap.version == latest.version else marker
        print(f"  v{snap.version:04d}: {snap.score:.1f} {snap.description[:60]}{marker}")


def cmd_status(args):
    """Show overall improvement status."""
    data_dir = Path("data/phase8")
    state_path = data_dir / "orchestrator_state.json"

    if state_path.exists():
        state = json.loads(state_path.read_text())
        summary = state.get("summary", {})
        print("\n=== Self-Improvement Status ===\n")
        print(f"Sessions: {summary.get('total_sessions', 0)}")
        print(f"Converged: {summary.get('converged_sessions', 0)}")
        print(f"Total improvement: {summary.get('total_improvement', 0):+.1f}")
        print(f"Avg improvement/session: {summary.get('avg_improvement_per_session', 0):+.1f}")
        print(f"Best score: {summary.get('best_final_score', 0):.1f}")
        print(f"Skills learned: {summary.get('total_skills_learned', 0)}")
        print(f"Failures fixed: {summary.get('total_failures_fixed', 0)}")
        print(f"Latest status: {summary.get('latest_status', 'unknown')}")
    else:
        print("\nNo improvement data yet. Run an improvement session first.")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 8: Self-Improving AI CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s evaluate --test-command "pytest tests/"
  %(prog)s replay --task-type code_generation --outcome success
  %(prog)s failures -n 20
  %(prog)s skills -n 10
  %(prog)s adapters
  %(prog)s status
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Commands")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate current code quality")
    p_eval.add_argument("--test-command", "-t", help="Test command to run")
    p_eval.set_defaults(func=cmd_evaluate)

    # replay
    p_replay = sub.add_parser("replay", help="Show experience replay insights")
    p_replay.add_argument("--task-type", "-T", help="Filter by task type")
    p_replay.add_argument("--outcome", "-o", choices=["success", "failure", "partial"], help="Filter by outcome")
    p_replay.add_argument("-n", type=int, default=10, help="Number of samples")
    p_replay.add_argument("--best", action="store_true", help="Prioritize high scores")
    p_replay.add_argument("--no-recent", action="store_true", help="Don't prioritize recent")
    p_replay.set_defaults(func=cmd_replay)

    # failures
    p_fail = sub.add_parser("failures", help="Show failure database")
    p_fail.add_argument("-n", type=int, default=20, help="Number of patterns")
    p_fail.set_defaults(func=cmd_failures)

    # skills
    p_skills = sub.add_parser("skills", help="Show learned meta-skills")
    p_skills.add_argument("-n", type=int, default=10, help="Number of skills")
    p_skills.set_defaults(func=cmd_skills)

    # adapters
    p_adapt = sub.add_parser("adapters", help="Manage LoRA adapters")
    p_adapt.set_defaults(func=cmd_adapters)

    # status
    p_status = sub.add_parser("status", help="Show overall improvement status")
    p_status.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
