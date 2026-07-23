"""
Phase 9: AI Research Scientist — Autonomous scientific discovery.

This module implements an AI that can:
  - Read and understand AI research papers (arXiv client + parser)
  - Design and run experiments automatically
  - Propose novel algorithms by combining known techniques
  - Form and test scientific hypotheses with statistical rigor
  - Write academic-style research reports (Markdown + LaTeX)
  - Coordinate the full research pipeline through a unified orchestrator
"""

from .paper_reader import (
    ArXivClient,
    PaperParser,
    Paper,
    PaperSection,
    Algorithm,
    ResearchLibrary,
    ResearchGraph,
)

from .experiment_runner import (
    ExperimentDesigner,
    ExperimentRunner,
    ExperimentTracker,
    ExperimentConfig,
    ExperimentResult,
    ExperimentComparison,
    ExperimentStatus,
)

from .algorithm_proposer import (
    AlgorithmProposer,
    AlgorithmProposal,
    ProposalStatus,
    NoveltyDetector,
    FeasibilityEstimator,
)

from .hypothesis_tester import (
    HypothesisTester,
    HypothesisFormulator,
    StatisticalAnalyzer,
    Hypothesis,
    HypothesisStatus,
    HypothesisTestResult,
    ConfidenceLevel,
)

from .report_writer import (
    ReportWriter,
    ResearchReport,
    ExperimentSummary,
    ReportSection,
)

from .orchestrator import (
    Phase9Orchestrator,
    ResearchSession,
)

__all__ = [
    # Paper reader
    "ArXivClient", "PaperParser", "Paper", "PaperSection",
    "Algorithm", "ResearchLibrary", "ResearchGraph",
    # Experiment runner
    "ExperimentDesigner", "ExperimentRunner", "ExperimentTracker",
    "ExperimentConfig", "ExperimentResult", "ExperimentComparison",
    "ExperimentStatus",
    # Algorithm proposer
    "AlgorithmProposer", "AlgorithmProposal", "ProposalStatus",
    "NoveltyDetector", "FeasibilityEstimator",
    # Hypothesis tester
    "HypothesisTester", "HypothesisFormulator", "StatisticalAnalyzer",
    "Hypothesis", "HypothesisStatus", "HypothesisTestResult",
    "ConfidenceLevel",
    # Report writer
    "ReportWriter", "ResearchReport", "ExperimentSummary", "ReportSection",
    # Orchestrator
    "Phase9Orchestrator", "ResearchSession",
]

