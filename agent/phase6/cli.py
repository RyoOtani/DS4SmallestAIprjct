#!/usr/bin/env python3
"""
Phase 6 CLI — AI Software Engineer Professional.
Usage:
  python -m agent.phase6.cli scan <workspace>
  python -m agent.phase6.cli analyze <workspace>
  python -m agent.phase6.cli review <file>
  python -m agent.phase6.cli quality <workspace>
  python -m agent.phase6.cli symbol <name> --workspace <path>
  python -m agent.phase6.cli full <workspace> --task "Add feature X"
"""

import argparse
import sys
from pathlib import Path

# Ensure tinyllm root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from agent.phase6.orchestrator import Phase6Orchestrator
from agent.phase6.code_understanding import RepositoryAnalyzer
from agent.phase6.architect import ArchitectAgent
from agent.phase6.critic import CriticAgent
from agent.phase6.quality import QualityPipeline


def cmd_scan(args):
    """Scan a repository for deep code understanding."""
    orch = Phase6Orchestrator(args.workspace)
    result = orch.scan_repository()
    print(f"✓ Scanned: {result['files']} files, {result['symbols']} symbols, "
          f"{result['functions']} functions, {result['classes']} classes")
    return 0


def cmd_analyze(args):
    """Analyze repository architecture."""
    orch = Phase6Orchestrator(args.workspace)
    orch.scan_repository()
    result = orch.analyze_architecture()
    print(f"Components: {result['components']}")
    print(f"Patterns: {result['patterns']}")
    print(f"Risks ({len(result['risks'])}):")
    for r in result['risks']:
        print(f"  ⚠ {r}")
    print(f"Suggestions ({len(result['suggestions'])}):")
    for s in result['suggestions']:
        print(f"  → [{s['type']}] {s['reason']}")
    return 0


def cmd_review(args):
    """Review a file or directory."""
    critic = CriticAgent()
    if Path(args.target).is_dir():
        reviews = critic.review_directory(args.target)
        for fpath, review in reviews.items():
            print(f"\n{'='*60}")
            print(f"File: {fpath}")
            print(f"Score: {review.score}/10  |  Blockers: {len(review.blockers)}  |  Passed: {review.passed}")
            for c in review.blockers[:5]:
                print(f"  🔴 [{c.category}] line {c.line}: {c.message}")
            for c in [x for x in review.comments if x.severity == "major"][:3]:
                print(f"  🟡 [{c.category}] line {c.line}: {c.message}")
    else:
        review = critic.review_file(args.target)
        print(f"Score: {review.score}/10  |  Blockers: {len(review.blockers)}  |  Passed: {review.passed}")
        for c in review.comments[:15]:
            emoji = {"blocker": "🔴", "major": "🟡", "minor": "🔵"}.get(c.severity, "⚪")
            print(f"  {emoji} [{c.category}] line {c.line}: {c.message}")
            if c.suggestion:
                print(f"     ↳ Suggest: {c.suggestion}")
    return 0


def cmd_quality(args):
    """Run quality pipeline."""
    pipeline = QualityPipeline()
    result = pipeline.run_full_pipeline(args.workspace, auto_fix=args.fix)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_symbol(args):
    """Find a symbol in the repository."""
    analyzer = RepositoryAnalyzer(args.workspace or ".")
    analyzer.scan()
    result = analyzer.find_symbol_usages(args.name)
    if result:
        for s in result:
            print(f"{s.kind:10s} {s.name:30s} {s.file}:{s.line}")
            if s.signature:
                print(f"           {s.signature}")
    else:
        print(f"Symbol '{args.name}' not found.")
    callers = analyzer.find_callers(args.name)
    if callers:
        print(f"\nCallers ({len(callers)}):")
        for c in callers[:10]:
            print(f"  {c.caller} → {c.callee}  ({c.caller_file})")
    return 0


def cmd_full(args):
    """Run full Phase 6 pipeline."""
    orch = Phase6Orchestrator(args.workspace)
    print(f"Running full pipeline on: {args.workspace}")
    print(f"Task: {args.task}")
    print()
    
    result = orch.run_full_cycle(
        task=args.task,
        build_cmd=args.build_cmd or "",
        test_cmd=args.test_cmd or "",
    )
    
    print(json.dumps(result["overall"], indent=2, ensure_ascii=False))
    print()
    print(orch.generate_report())
    return 0 if result["overall"]["status"] == "ready" else 2


def main():
    parser = argparse.ArgumentParser(description="Phase 6: AI Software Engineer Professional")
    sub = parser.add_subparsers(dest="command")
    
    scan = sub.add_parser("scan", help="Scan repository for deep understanding")
    scan.add_argument("workspace")
    
    analyze = sub.add_parser("analyze", help="Analyze repository architecture")
    analyze.add_argument("workspace")
    
    review = sub.add_parser("review", help="Review code quality")
    review.add_argument("target", help="File or directory to review")
    
    quality = sub.add_parser("quality", help="Run quality pipeline")
    quality.add_argument("workspace")
    quality.add_argument("--fix", action="store_true", help="Auto-format files")
    
    symbol = sub.add_parser("symbol", help="Find a symbol in the repository")
    symbol.add_argument("name")
    symbol.add_argument("--workspace", default=".", help="Repository path")
    
    full = sub.add_parser("full", help="Run full AI engineer pipeline")
    full.add_argument("workspace")
    full.add_argument("--task", required=True, help="Task description")
    full.add_argument("--build-cmd", default="", help="Build command")
    full.add_argument("--test-cmd", default="", help="Test command")
    
    args = parser.parse_args()
    
    commands = {
        "scan": cmd_scan,
        "analyze": cmd_analyze,
        "review": cmd_review,
        "quality": cmd_quality,
        "symbol": cmd_symbol,
        "full": cmd_full,
    }
    
    if args.command in commands:
        return commands[args.command](args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
