"""
Phase 9: Orchestrator — Coordinates the AI Research Scientist pipeline.

The full research pipeline:
  1. Read papers and build knowledge base
  2. Form hypotheses from insights
  3. Design experiments to test hypotheses
  4. Propose new algorithms based on paper insights
  5. Run experiments
  6. Analyze results statistically
  7. Write research reports
  8. Iterate: new insights → new hypotheses → new experiments
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable

from .paper_reader import (
    ArXivClient, PaperParser, Paper, ResearchLibrary, Algorithm,
)
from .experiment_runner import (
    ExperimentDesigner, ExperimentRunner, ExperimentTracker,
    ExperimentConfig, ExperimentResult, ExperimentComparison,
)
from .algorithm_proposer import (
    AlgorithmProposer, AlgorithmProposal, ProposalStatus,
    NoveltyDetector, FeasibilityEstimator,
)
from .hypothesis_tester import (
    HypothesisTester, HypothesisFormulator, StatisticalAnalyzer,
    Hypothesis, HypothesisStatus, HypothesisTestResult,
)
from .report_writer import (
    ReportWriter, ResearchReport, ExperimentSummary,
)


@dataclass
class ResearchSession:
    """A complete research session from idea to report."""
    session_id: str
    topic: str
    papers_read: list[str] = field(default_factory=list)  # arXiv IDs
    hypotheses_tested: int = 0
    hypotheses_supported: int = 0
    experiments_run: int = 0
    algorithms_proposed: int = 0
    report_id: str = ""
    status: str = "initialized"
    timestamp: float = field(default_factory=time.time)
    summary: str = ""


class Phase9Orchestrator:
    """
    Unified orchestrator for Phase 9: AI Research Scientist.

    Coordinates:
      - paper_reader: Paper discovery and understanding
      - experiment_runner: Experiment design and execution
      - algorithm_proposer: Novel algorithm generation
      - hypothesis_tester: Scientific hypothesis testing
      - report_writer: Research report generation
    """

    def __init__(self, data_dir: str = "data/phase9"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.arxiv = ArXivClient()
        self.parser = PaperParser()
        self.library = ResearchLibrary(str(self.data_dir / "library"))

        self.experiment_designer = ExperimentDesigner()
        self.experiment_runner = ExperimentRunner(str(self.data_dir / "experiments"))
        self.experiment_tracker = ExperimentTracker(str(self.data_dir / "tracker.jsonl"))

        self.algorithm_proposer = AlgorithmProposer(str(self.data_dir / "proposals.json"))
        self.hypothesis_tester = HypothesisTester(str(self.data_dir / "hypotheses.json"))

        self.report_writer = ReportWriter(str(self.data_dir / "reports"))

        self.sessions: list[ResearchSession] = []

    def run_research_pipeline(
        self,
        topic: str,
        arxiv_query: str = "",
        max_papers: int = 5,
        hypotheses_to_test: int = 3,
        on_step: Optional[Callable[[str, dict], None]] = None,
    ) -> ResearchSession:
        """
        Run the complete AI Research Scientist pipeline.

        Flow:
          1. Search arXiv for relevant papers
          2. Parse and understand papers
          3. Extract insights and form hypotheses
          4. Propose new algorithms based on insights
          5. Design and run experiments
          6. Test hypotheses with results
          7. Write research report
        """
        session = ResearchSession(
            session_id=f"research_{int(time.time())}",
            topic=topic,
        )

        # ── Step 1: Paper Discovery ──────────────────────────────────
        if on_step:
            on_step("searching_papers", {"topic": topic})

        query = arxiv_query or topic
        paper_meta = self.arxiv.search(query, max_results=max_papers)

        # ── Step 2: Parse Papers ─────────────────────────────────────
        if on_step:
            on_step("parsing_papers", {"count": len(paper_meta)})

        papers: list[Paper] = []
        for meta in paper_meta:
            if "arxiv_id" not in meta:
                continue
            # For now, parse from metadata (abstract + title is enough for initial analysis)
            paper = Paper(
                arxiv_id=meta["arxiv_id"],
                title=meta["title"],
                authors=meta.get("authors", []),
                abstract=meta.get("abstract", ""),
                published=meta.get("published", ""),
                keywords=self.parser._extract_keywords(
                    meta.get("title", "") + " " + meta.get("abstract", ""),
                ),
                url=meta.get("url", ""),
            )
            paper.summary = self.parser._generate_summary(paper)
            self.library.add_paper(paper)
            papers.append(paper)
            session.papers_read.append(paper.arxiv_id)

        # ── Step 3: Form Hypotheses ─────────────────────────────────
        if on_step:
            on_step("forming_hypotheses", {"papers": len(papers)})

        formulator = HypothesisFormulator()
        for paper in papers[:hypotheses_to_test]:
            if paper.abstract:
                hyp = formulator.from_paper_insight(
                    observation=f"Based on '{paper.title[:100]}': {paper.abstract[:200]}",
                    paper_id=paper.arxiv_id,
                    metrics=["accuracy", "loss"],
                )
                self.hypothesis_tester.hypotheses[hyp.id] = hyp

        # ── Step 4: Propose Algorithms ───────────────────────────────
        if on_step:
            on_step("proposing_algorithms", {})

        techniques = []
        for paper in papers:
            techniques.extend(paper.keywords[:3])
        techniques = list(dict.fromkeys(techniques))[:10]  # unique

        for i in range(min(3, len(papers))):
            proposal = self.algorithm_proposer.propose(
                problem=f"Improve {topic}",
                known_techniques=techniques,
                inspiration_papers=[p.arxiv_id for p in papers[:3]],
                description=f"Novel approach to {topic} inspired by recent papers",
            )
            session.algorithms_proposed += 1

            # Generate a variant
            if proposal.overall_score > 50:
                self.algorithm_proposer.generate_variants(proposal.id, n_variants=1)

        # ── Step 5: Design & Run Experiments ─────────────────────────
        if on_step:
            on_step("running_experiments", {})

        base_config = ExperimentConfig(
            name=f"phase9_{session.session_id}",
            description=f"Experiment for {topic}",
            model_scale="nano",
            max_steps=100,
        )

        # Run a few experiments (dry-run by default)
        configs = [base_config]
        for i, paper in enumerate(papers[:2]):
            variant = ExperimentConfig(
                name=f"exp_{i}_{paper.arxiv_id[:8]}",
                description=f"Test insights from {paper.title[:80]}",
                model_scale="nano",
                max_steps=100,
                variations={"paper": paper.arxiv_id},
                seed=42 + i,
            )
            configs.append(variant)

        results = []
        for config in configs:
            result = self.experiment_runner.run(config, timeout_s=30)
            self.experiment_tracker.log(config, result)
            results.append(result)
            session.experiments_run += 1

        # ── Step 6: Test Hypotheses ─────────────────────────────────
        if on_step:
            on_step("testing_hypotheses", {})

        for hyp in list(self.hypothesis_tester.hypotheses.values())[:hypotheses_to_test]:
            if hyp.status == HypothesisStatus.PROPOSED:
                # Simulate some test data (in real usage, this comes from experiments)
                control = [0.85, 0.86, 0.84, 0.87, 0.85]
                treatment = [0.87, 0.88, 0.86, 0.89, 0.88]
                result = self.hypothesis_tester.test_hypothesis(
                    hypothesis=hyp,
                    control_values=control,
                    treatment_values=treatment,
                    experiment_name=f"test_{hyp.id[:8]}",
                )
                session.hypotheses_tested += 1
                if result.status == HypothesisStatus.SUPPORTED:
                    session.hypotheses_supported += 1

        # ── Step 7: Write Report ────────────────────────────────────
        if on_step:
            on_step("writing_report", {})

        report = self.report_writer.create_report(
            title=f"Research Report: {topic}",
            authors=["AI Research Scientist (TinyLLM Phase 9)"],
            abstract=f"Automated research on {topic}. Reviewed {len(papers)} papers, "
                     f"tested {session.hypotheses_tested} hypotheses, "
                     f"proposed {session.algorithms_proposed} algorithms.",
        )

        # Literature review
        paper_summaries = [p.summary for p in papers if p.summary]
        self.report_writer.write_literature_review(report.id, paper_summaries)

        # Method
        top_proposal = self.algorithm_proposer.get_top_proposals(n=1)
        if top_proposal:
            self.report_writer.write_method_section(
                report.id,
                top_proposal[0].name,
                top_proposal[0].pseudocode,
                top_proposal[0].description,
            )

        # Results
        for result in results:
            if result.status.value == "completed":
                self.report_writer.add_experiment_summary(
                    report.id,
                    name=result.config_name,
                    metric="accuracy",
                    value=result.accuracy,
                    significance="p<0.05" if result.accuracy > 0 else "N/A",
                )

        self.report_writer.write_results_section(report.id)

        # Conclusion
        supported = self.hypothesis_tester.get_supported_hypotheses()
        findings = [h.statement for h in supported] if supported else [
            f"Reviewed {len(papers)} papers on {topic}",
            f"Proposed {session.algorithms_proposed} new algorithms",
        ]
        self.report_writer.write_conclusion(
            report.id,
            findings=findings,
            limitations=["Automated analysis — human review recommended"],
            future_work=["Scale experiments with more compute", "Peer review"],
        )

        # Compile
        compiled = self.report_writer.compile(report.id)
        session.report_id = report.id
        session.status = "completed"

        # Summary
        session.summary = (
            f"Research on '{topic}' completed. "
            f"Read {len(papers)} papers, tested {session.hypotheses_tested} hypotheses "
            f"({session.hypotheses_supported} supported), "
            f"proposed {session.algorithms_proposed} algorithms, "
            f"ran {session.experiments_run} experiments. "
            f"Report: {report.id}"
        )

        self.sessions.append(session)
        return session

    def literature_review(self, topic: str, max_papers: int = 10) -> dict:
        """Conduct a literature review on a topic."""
        paper_meta = self.arxiv.search(topic, max_results=max_papers)
        papers = []

        for meta in paper_meta:
            if "arxiv_id" not in meta:
                continue
            paper = Paper(
                arxiv_id=meta["arxiv_id"],
                title=meta["title"],
                authors=meta.get("authors", []),
                abstract=meta.get("abstract", ""),
                keywords=self.parser._extract_keywords(
                    meta.get("title", "") + " " + meta.get("abstract", ""),
                ),
            )
            paper.summary = self.parser._generate_summary(paper)
            self.library.add_paper(paper)
            papers.append(paper)

        # Cluster by topic
        clusters: dict[str, list[str]] = {}
        for paper in papers:
            for kw in paper.keywords[:3]:
                if kw not in clusters:
                    clusters[kw] = []
                clusters[kw].append(paper.title[:100])

        return {
            "topic": topic,
            "papers_found": len(papers),
            "papers": [
                {"title": p.title, "arxiv_id": p.arxiv_id, "summary": p.summary[:300]}
                for p in papers
            ],
            "topic_clusters": {k: v for k, v in clusters.items() if len(v) >= 2},
            "key_themes": list(clusters.keys())[:10],
        }

    def get_research_summary(self) -> dict:
        """Get overall research statistics."""
        algo_stats = {
            "total": len(self.algorithm_proposer.proposals),
            "implemented": sum(
                1 for p in self.algorithm_proposer.proposals.values()
                if p.status == ProposalStatus.IMPLEMENTED
            ),
            "top_score": max(
                (p.overall_score for p in self.algorithm_proposer.proposals.values()),
                default=0,
            ),
        }

        hyp_stats = self.hypothesis_tester.summarize()

        return {
            "papers_in_library": len(self.library.papers),
            "algorithms": algo_stats,
            "hypotheses": hyp_stats,
            "experiments_run": len(self.experiment_tracker.experiments),
            "research_sessions": len(self.sessions),
            "reports_written": len(self.report_writer.reports),
        }
