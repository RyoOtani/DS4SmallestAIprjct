#!/usr/bin/env python3
"""
Phase 9 CLI: AI Research Scientist command-line interface.

Usage:
  python -m agent.phase9.cli search       Search arXiv for papers
  python -m agent.phase9.cli read         Read and parse a paper
  python -m agent.phase9.cli library      Manage paper library
  python -m agent.phase9.cli propose      Propose a new algorithm
  python -m agent.phase9.cli experiment   Design and run experiments
  python -m agent.phase9.cli hypothesis   Form and test hypotheses
  python -m agent.phase9.cli report       Generate research reports
  python -m agent.phase9.cli research     Run full research pipeline
  python -m agent.phase9.cli status       Show research summary
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.phase9 import (
    ArXivClient,
    PaperParser,
    ResearchLibrary,
    ExperimentDesigner,
    ExperimentRunner,
    ExperimentConfig,
    AlgorithmProposer,
    HypothesisTester,
    HypothesisFormulator,
    ReportWriter,
    Phase9Orchestrator,
)


def cmd_search(args):
    """Search arXiv for papers."""
    client = ArXivClient()
    results = client.search(
        query=args.query,
        max_results=args.max,
        sort_by=args.sort,
    )

    print(f"\n=== arXiv Search: '{args.query}' ({len(results)} results) ===\n")
    for i, paper in enumerate(results, 1):
        if "error" in paper:
            print(f"Error: {paper['error']}")
            continue
        print(f"{i}. {paper.get('title', 'N/A')[:120]}")
        authors = ", ".join(paper.get("authors", [])[:5])
        print(f"   Authors: {authors}")
        print(f"   ID: {paper.get('arxiv_id', 'N/A')}")
        print(f"   Published: {paper.get('published', 'N/A')[:10]}")
        print(f"   Abstract: {paper.get('abstract', '')[:200]}...")
        print()


def cmd_read(args):
    """Read and parse a specific paper."""
    client = ArXivClient()
    parser = PaperParser()

    meta = client.get_paper(args.arxiv_id)
    if not meta:
        print(f"Paper not found: {args.arxiv_id}")
        return

    # Parse from metadata
    text = f"{meta['title']}\n\n{meta['abstract']}"
    paper = parser.parse_paper_text(text, arxiv_id=meta["arxiv_id"])

    print(f"\n=== Paper: {paper.title} ===\n")
    print(f"Authors: {', '.join(paper.authors[:10])}")
    print(f"Published: {paper.published}")
    print(f"\n--- Abstract ---\n{paper.abstract[:500]}")
    print(f"\n--- Keywords ---\n{', '.join(paper.keywords[:20])}")
    if paper.contributions:
        print(f"\n--- Contributions ---")
        for c in paper.contributions:
            print(f"  • {c}")
    if paper.algorithms:
        print(f"\n--- Algorithms ({len(paper.algorithms)}) ---")
        for alg in paper.algorithms:
            print(f"  • {alg.name}: {alg.pseudocode[:200]}")
    print(f"\n--- Summary ---\n{paper.summary}")


def cmd_library(args):
    """Manage the paper library."""
    lib = ResearchLibrary()

    if args.action == "list":
        print(f"\n=== Paper Library ({len(lib.papers)} papers) ===\n")
        for pid, paper in lib.papers.items():
            print(f"📄 [{pid}] {paper.title[:100]}")
            print(f"   Keywords: {', '.join(paper.keywords[:8])}")
            print()

    elif args.action == "search":
        results = lib.search_library(args.query, field=args.field)
        print(f"\n=== Library Search: '{args.query}' ({len(results)} results) ===\n")
        for paper in results:
            print(f"📄 [{paper.arxiv_id}] {paper.title[:100]}")
            print(f"   {paper.abstract[:150]}...")
            print()

    elif args.action == "related":
        if not args.arxiv_id:
            print("Specify --arxiv-id")
            return
        related = lib.get_related(args.arxiv_id)
        print(f"\n=== Related to {args.arxiv_id} ({len(related)} papers) ===\n")
        for paper in related[:10]:
            print(f"📄 [{paper.arxiv_id}] {paper.title[:100]}")


def cmd_propose(args):
    """Propose a new algorithm."""
    proposer = AlgorithmProposer()
    proposal = proposer.propose(
        problem=args.problem,
        known_techniques=args.techniques.split(",") if args.techniques else [],
        inspiration_papers=[],
        description=args.description or "",
    )

    print(f"\n=== Algorithm Proposal: {proposal.name} ===\n")
    print(f"Problem: {proposal.problem_statement[:200]}")
    print(f"Core Idea: {proposal.core_idea[:200]}")
    print(f"Building Blocks: {', '.join(proposal.building_blocks)}")
    print(f"\nScores (0-100):")
    print(f"  Novelty:    {proposal.novelty_score:.0f}")
    print(f"  Feasibility:{proposal.feasibility_score:.0f}")
    print(f"  Impact:     {proposal.impact_score:.0f}")
    print(f"  Overall:    {proposal.overall_score:.0f}")

    # Top proposals
    top = proposer.get_top_proposals(n=5)
    if top:
        print(f"\n--- Top Proposals ---")
        for p in top:
            print(f"  {p.overall_score:.0f} | {p.name[:70]}")

    if args.variants:
        variants = proposer.generate_variants(proposal.id, n_variants=3)
        print(f"\n--- Generated {len(variants)} Variants ---")
        for v in variants:
            print(f"  {v.overall_score:.0f} | {v.name[:70]}")


def cmd_experiment(args):
    """Design and run experiments."""
    designer = ExperimentDesigner()
    runner = ExperimentRunner()

    base = ExperimentConfig(
        name=args.name or "experiment",
        description=args.description or "Phase 9 experiment",
        model_scale=args.scale,
        learning_rate=args.lr,
        max_steps=args.steps,
        seed=args.seed,
    )

    configs = [base]

    if args.grid:
        # Simple grid: learning rate sweep
        configs = designer.design_grid_search(
            base,
            {"learning_rate": [1e-5, 5e-5, 1e-4, 5e-4]},
        )

    if args.ablation:
        configs = designer.design_ablation(base, args.ablation.split(","))

    print(f"\n=== Running {len(configs)} experiment(s) ===\n")
    results = runner.run_batch(configs, timeout_s=args.timeout)

    comparison = runner.compare(results)
    print(comparison.summary)

    for result in results:
        status_emoji = "✅" if result.status.value == "completed" else "❌"
        print(f"\n{status_emoji} {result.config_name}")
        print(f"   Status: {result.status.value}")
        print(f"   Metrics: {result.metrics}")
        print(f"   Runtime: {result.runtime_s}s")
        if result.warnings:
            for w in result.warnings:
                print(f"   ⚠️  {w}")


def cmd_hypothesis(args):
    """Form and test hypotheses."""
    tester = HypothesisTester()
    formulator = HypothesisFormulator()

    if args.action == "form":
        hyp = formulator.from_observation(
            observation=args.observation,
            metrics=args.metrics.split(",") if args.metrics else ["accuracy"],
        )
        print(f"\n=== Hypothesis Formed ===\n")
        print(f"ID: {hyp.id}")
        print(f"H₀: {hyp.null_hypothesis}")
        print(f"H₁: {hyp.alternative_hypothesis}")
        print(f"Metrics: {', '.join(hyp.metrics_to_measure)}")

    elif args.action == "test":
        # Use provided or sample data
        control = [float(x) for x in args.control.split(",")] if args.control else [0.85, 0.86, 0.84]
        treatment = [float(x) for x in args.treatment.split(",")] if args.treatment else [0.88, 0.89, 0.87]

        hyp = Hypothesis(
            id="manual_test",
            statement=args.observation or "Manual hypothesis test",
            null_hypothesis="No difference",
            alternative_hypothesis="Significant difference",
            metrics_to_measure=["accuracy"],
        )

        result = tester.test_hypothesis(
            hypothesis=hyp,
            control_values=control,
            treatment_values=treatment,
            experiment_name=args.name or "manual_test",
            alpha=args.alpha,
        )

        print(f"\n=== Hypothesis Test Result ===\n")
        print(f"Status: {result.status.value}")
        print(f"p-value: {result.p_value:.6f}")
        print(f"Effect size (Cohen's d): {result.effect_size:.3f}")
        print(f"Control mean: {result.control_mean:.4f}")
        print(f"Treatment mean: {result.treatment_mean:.4f}")
        print(f"Delta: {result.delta:+.4f}")
        print(f"Conclusion: {result.conclusion}")

    elif args.action == "summary":
        summary = tester.summarize()
        print(f"\n=== Hypothesis Testing Summary ===\n")
        for k, v in summary.items():
            print(f"  {k}: {v}")


def cmd_report(args):
    """Generate research reports."""
    writer = ReportWriter()

    if args.action == "create":
        report = writer.create_report(
            title=args.title,
            authors=args.authors.split(",") if args.authors else ["AI Scientist"],
            abstract=args.abstract or "",
        )
        print(f"\n=== Report Created ===\n")
        print(f"ID: {report.id}")
        print(f"Title: {report.title}")
        print(f"Version: {report.version}")

        if args.compile:
            compiled = writer.compile(report.id)
            if compiled:
                print(f"\nCompiled: {report.compiled_path}")
                print(f"\n--- Preview (first 500 chars) ---")
                print(compiled[:500])

    elif args.action == "list":
        print(f"\n=== Reports ({len(writer.reports)}) ===\n")
        for rid, report in writer.reports.items():
            print(f"📝 [{rid[:8]}] {report.title[:80]} (v{report.version})")


def cmd_research(args):
    """Run the full research pipeline."""
    orch = Phase9Orchestrator(data_dir=args.data_dir)

    def on_step(step: str, info: dict):
        print(f"  [{step}] {info}")

    print(f"\n🔬 Running AI Research Pipeline on '{args.topic}'...\n")
    session = orch.run_research_pipeline(
        topic=args.topic,
        arxiv_query=args.query or args.topic,
        max_papers=args.max_papers,
        hypotheses_to_test=args.hypotheses,
        on_step=on_step,
    )

    print(f"\n=== Research Complete ===\n")
    print(f"Session: {session.session_id}")
    print(f"Papers read: {len(session.papers_read)}")
    print(f"Hypotheses tested: {session.hypotheses_tested} ({session.hypotheses_supported} supported)")
    print(f"Experiments run: {session.experiments_run}")
    print(f"Algorithms proposed: {session.algorithms_proposed}")
    print(f"Report ID: {session.report_id}")
    print(f"\nSummary: {session.summary}")

    # Print research summary
    summary = orch.get_research_summary()
    print(f"\n--- Overall Research Stats ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")


def cmd_status(args):
    """Show research summary status."""
    orch = Phase9Orchestrator(data_dir=args.data_dir)

    summary = orch.get_research_summary()
    print(f"\n=== AI Research Scientist Status ===\n")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    top_proposals = orch.algorithm_proposer.get_top_proposals(n=3)
    if top_proposals:
        print(f"\n--- Top Algorithm Proposals ---")
        for p in top_proposals:
            print(f"  [{p.overall_score:.0f}] {p.name[:80]}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 9: AI Research Scientist CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s search --query "mixture of experts transformer" --max 5
  %(prog)s read --arxiv-id 1706.03762
  %(prog)s library list
  %(prog)s library search --query "attention"
  %(prog)s propose --problem "efficient LLM inference" --techniques "MoE,MLA,distillation"
  %(prog)s experiment --name "lr_sweep" --grid
  %(prog)s hypothesis form --observation "MLA reduces KV cache without quality loss"
  %(prog)s hypothesis test --control "0.85,0.86,0.84" --treatment "0.88,0.89,0.87"
  %(prog)s report create --title "Efficient Attention Survey" --compile
  %(prog)s research --topic "efficient transformer attention"
  %(prog)s status
        """,
    )

    sub = parser.add_subparsers(dest="command", help="Commands")

    # search
    p_search = sub.add_parser("search", help="Search arXiv for papers")
    p_search.add_argument("--query", "-q", required=True, help="Search query")
    p_search.add_argument("--max", type=int, default=10)
    p_search.add_argument("--sort", default="relevance")
    p_search.set_defaults(func=cmd_search)

    # read
    p_read = sub.add_parser("read", help="Read and parse a paper")
    p_read.add_argument("--arxiv-id", "-i", required=True)
    p_read.set_defaults(func=cmd_read)

    # library
    p_lib = sub.add_parser("library", help="Manage paper library")
    p_lib.add_argument("action", nargs="?", default="list",
                       choices=["list", "search", "related"])
    p_lib.add_argument("--query", "-q")
    p_lib.add_argument("--field", "-f", default="all")
    p_lib.add_argument("--arxiv-id", "-i")
    p_lib.set_defaults(func=cmd_library)

    # propose
    p_prop = sub.add_parser("propose", help="Propose a new algorithm")
    p_prop.add_argument("--problem", "-p", required=True)
    p_prop.add_argument("--techniques", "-t")
    p_prop.add_argument("--description", "-d")
    p_prop.add_argument("--variants", action="store_true")
    p_prop.set_defaults(func=cmd_propose)

    # experiment
    p_exp = sub.add_parser("experiment", help="Design and run experiments")
    p_exp.add_argument("--name", "-n")
    p_exp.add_argument("--description", "-d")
    p_exp.add_argument("--scale", default="nano")
    p_exp.add_argument("--lr", type=float, default=1e-4)
    p_exp.add_argument("--steps", type=int, default=100)
    p_exp.add_argument("--seed", type=int, default=42)
    p_exp.add_argument("--timeout", type=int, default=60)
    p_exp.add_argument("--grid", action="store_true")
    p_exp.add_argument("--ablation")
    p_exp.set_defaults(func=cmd_experiment)

    # hypothesis
    p_hyp = sub.add_parser("hypothesis", help="Form and test hypotheses")
    p_hyp.add_argument("action", nargs="?", default="summary",
                       choices=["form", "test", "summary"])
    p_hyp.add_argument("--observation", "-o")
    p_hyp.add_argument("--metrics", "-m")
    p_hyp.add_argument("--control")
    p_hyp.add_argument("--treatment")
    p_hyp.add_argument("--name", "-n")
    p_hyp.add_argument("--alpha", type=float, default=0.05)
    p_hyp.set_defaults(func=cmd_hypothesis)

    # report
    p_rep = sub.add_parser("report", help="Generate research reports")
    p_rep.add_argument("action", nargs="?", default="list",
                       choices=["create", "list"])
    p_rep.add_argument("--title", "-t")
    p_rep.add_argument("--authors", "-a")
    p_rep.add_argument("--abstract")
    p_rep.add_argument("--compile", action="store_true")
    p_rep.set_defaults(func=cmd_report)

    # research (full pipeline)
    p_res = sub.add_parser("research", help="Run full research pipeline")
    p_res.add_argument("--topic", "-t", required=True)
    p_res.add_argument("--query", "-q")
    p_res.add_argument("--max-papers", type=int, default=5)
    p_res.add_argument("--hypotheses", type=int, default=3)
    p_res.add_argument("--data-dir", default="data/phase9")
    p_res.set_defaults(func=cmd_research)

    # status
    p_stat = sub.add_parser("status", help="Show research summary")
    p_stat.add_argument("--data-dir", default="data/phase9")
    p_stat.set_defaults(func=cmd_status)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
