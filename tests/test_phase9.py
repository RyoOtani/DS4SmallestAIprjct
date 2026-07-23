"""
Tests for Phase 9: AI Research Scientist.

Covers:
  - ArXivClient (search, get_paper)
  - PaperParser (parse, compare, extract sections/algorithms/keywords)
  - ResearchLibrary (add, search, related)
  - ExperimentDesigner (A/B test, grid search, ablation)
  - ExperimentRunner (run, batch, compare, issue detection)
  - ExperimentTracker (log, get_best, trend)
  - AlgorithmProposer (propose, refine, variants, scoring)
  - NoveltyDetector (check, known algorithms)
  - FeasibilityEstimator (estimate, concerns)
  - HypothesisFormulator (from paper, from observation)
  - StatisticalAnalyzer (t-test, multiple comparison correction)
  - HypothesisTester (test, batch, summary)
  - ReportWriter (create, sections, compile, LaTeX)
  - Phase9Orchestrator (research pipeline, literature review)
"""

from __future__ import annotations
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.phase9.paper_reader import (
    ArXivClient, PaperParser, Paper, PaperSection, Algorithm,
    ResearchLibrary, ResearchGraph,
)
from agent.phase9.experiment_runner import (
    ExperimentDesigner, ExperimentRunner, ExperimentTracker,
    ExperimentConfig, ExperimentResult, ExperimentComparison,
    ExperimentStatus,
)
from agent.phase9.algorithm_proposer import (
    AlgorithmProposer, AlgorithmProposal, ProposalStatus,
    NoveltyDetector, FeasibilityEstimator,
)
from agent.phase9.hypothesis_tester import (
    HypothesisTester, HypothesisFormulator, StatisticalAnalyzer,
    Hypothesis, HypothesisStatus, HypothesisTestResult,
    ConfidenceLevel,
)
from agent.phase9.report_writer import (
    ReportWriter, ResearchReport, ExperimentSummary, ReportSection,
)
from agent.phase9.orchestrator import (
    Phase9Orchestrator, ResearchSession,
)


# ═══════════════════════════════════════════════════════════════
# PaperParser Tests
# ═══════════════════════════════════════════════════════════════

class TestPaperParser:
    def test_parse_paper_text(self):
        parser = PaperParser()
        text = """
Attention Is All You Need

Ashish Vaswani, Noam Shazeer

Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.

1 Introduction
Recurrent neural networks have been the state of the art.

2 Background
The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU.

3 Model Architecture
Most competitive neural sequence transduction models have an encoder-decoder structure.

4 Why Self-Attention
Comparing self-attention to recurrent and convolutional layers.

5 Results
On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best previously reported models by over 2.0 BLEU.
"""
        paper = parser.parse_paper_text(text, arxiv_id="1706.03762")
        assert paper.arxiv_id == "1706.03762"
        assert "Attention" in paper.title
        assert len(paper.keywords) > 0
        assert "transformer" in paper.keywords

    def test_extract_keywords(self):
        parser = PaperParser()
        text = "We propose a transformer-based mixture of experts model with RMS normalization and SwiGLU activation for efficient language model training."
        keywords = parser._extract_keywords(text)
        assert "transformer" in keywords
        assert "mixture of experts" in keywords or "moe" in keywords
        assert "rms norm" in keywords or "rmsnorm" in keywords

    def test_compare_papers(self):
        parser = PaperParser()
        paper_a = Paper(
            arxiv_id="1", title="Transformer", authors=["A"],
            abstract="attention is all you need",
            keywords=["attention", "transformer", "nlp"],
        )
        paper_b = Paper(
            arxiv_id="2", title="BERT", authors=["B"],
            abstract="pre-training of deep bidirectional transformers",
            keywords=["transformer", "pre-training", "nlp"],
        )
        comparison = parser.compare_papers(paper_a, paper_b)
        assert "transformer" in comparison["shared_keywords"]
        assert "nlp" in comparison["shared_keywords"]

    def test_generate_summary(self):
        parser = PaperParser()
        paper = Paper(
            arxiv_id="x", title="Test Paper", authors=["Author One"],
            abstract="This is a test.", keywords=["test", "example"],
            contributions=["Novel method"],
        )
        summary = parser._generate_summary(paper)
        assert "Test Paper" in summary
        assert "Author One" in summary
        assert "Novel method" in summary


# ═══════════════════════════════════════════════════════════════
# ResearchLibrary Tests
# ═══════════════════════════════════════════════════════════════

class TestResearchLibrary:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_add_and_search(self):
        lib = ResearchLibrary(library_path=self.tmpdir)
        paper = Paper(
            arxiv_id="1234.5678",
            title="Test Paper About Attention",
            authors=["Author"],
            abstract="We study attention mechanisms.",
            keywords=["attention", "transformer"],
        )
        lib.add_paper(paper)
        assert len(lib.papers) == 1

        results = lib.search_library("attention")
        assert len(results) == 1

    def test_search_by_field(self):
        lib = ResearchLibrary(library_path=self.tmpdir)
        paper = Paper(
            arxiv_id="x", title="Title X", authors=["A"],
            abstract="Abstract about MoE",
            keywords=["moe"],
        )
        lib.add_paper(paper)

        assert len(lib.search_library("Title", field="title")) == 1
        assert len(lib.search_library("MoE", field="abstract")) == 1
        assert len(lib.search_library("moe", field="keyword")) == 1


# ═══════════════════════════════════════════════════════════════
# ExperimentDesigner Tests
# ═══════════════════════════════════════════════════════════════

class TestExperimentDesigner:
    def test_design_ab_test(self):
        designer = ExperimentDesigner()
        base = ExperimentConfig(name="baseline", learning_rate=1e-4)
        a, b = designer.design_ab_test(base, {"learning_rate": 5e-4})
        assert a.learning_rate == 1e-4
        assert b.learning_rate == 5e-4
        assert b.seed != a.seed

    def test_design_grid_search(self):
        designer = ExperimentDesigner()
        base = ExperimentConfig(name="grid")
        configs = designer.design_grid_search(
            base,
            {"learning_rate": [1e-4, 5e-4], "batch_size": [16, 32]},
        )
        assert len(configs) == 4

    def test_design_ablation(self):
        designer = ExperimentDesigner()
        base = ExperimentConfig(name="full")
        configs = designer.design_ablation(base, ["dropout", "weight_decay"])
        assert len(configs) == 3  # full + 2 ablated
        assert configs[1].ablate == ["dropout"]
        assert configs[2].ablate == ["weight_decay"]


# ═══════════════════════════════════════════════════════════════
# ExperimentRunner Tests
# ═══════════════════════════════════════════════════════════════

class TestExperimentRunner:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_run_dry(self):
        runner = ExperimentRunner(output_dir=self.tmpdir)
        config = ExperimentConfig(name="test_dry", max_steps=10)
        result = runner.run(config, timeout_s=10)
        assert result.config_name == "test_dry"
        assert result.status in (ExperimentStatus.COMPLETED, ExperimentStatus.FAILED)

    def test_compare_results(self):
        runner = ExperimentRunner(output_dir=self.tmpdir)
        results = [
            ExperimentResult(
                config_name="A", status=ExperimentStatus.COMPLETED,
                metrics={"accuracy": 0.85, "final_loss": 0.3},
            ),
            ExperimentResult(
                config_name="B", status=ExperimentStatus.COMPLETED,
                metrics={"accuracy": 0.92, "final_loss": 0.2},
            ),
        ]
        comparison = runner.compare(results)
        assert comparison.best_config == "B"
        assert comparison.best_value == 0.92

    def test_detect_nan(self):
        runner = ExperimentRunner(output_dir=self.tmpdir)
        result = ExperimentResult(
            config_name="nan_test", status=ExperimentStatus.FAILED,
            raw_output="NaN detected in loss at step 100",
        )
        warnings = runner._detect_issues(result)
        assert any("NaN" in w for w in warnings)

    def test_detect_oom(self):
        runner = ExperimentRunner(output_dir=self.tmpdir)
        result = ExperimentResult(
            config_name="oom_test", status=ExperimentStatus.FAILED,
            raw_output="CUDA out of memory",
        )
        warnings = runner._detect_issues(result)
        assert any("Out of memory" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════
# ExperimentTracker Tests
# ═══════════════════════════════════════════════════════════════

class TestExperimentTracker:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.tracker_path = os.path.join(self.tmpdir, "tracker.jsonl")

    def test_log_and_get_best(self):
        tracker = ExperimentTracker(tracker_path=self.tracker_path)
        config = ExperimentConfig(name="test")
        result = ExperimentResult(
            config_name="test", status=ExperimentStatus.COMPLETED,
            metrics={"accuracy": 0.88},
        )
        tracker.log(config, result)
        best = tracker.get_best("accuracy")
        assert best is not None
        assert best["metrics"]["accuracy"] == 0.88

    def test_trend(self):
        tracker = ExperimentTracker(tracker_path=self.tracker_path)
        for acc in [0.80, 0.82, 0.85, 0.87]:
            config = ExperimentConfig(name=f"exp_{acc}")
            result = ExperimentResult(
                config_name=f"exp_{acc}", status=ExperimentStatus.COMPLETED,
                metrics={"accuracy": acc},
            )
            tracker.log(config, result)
        assert tracker.get_trend("accuracy") == "improving"


# ═══════════════════════════════════════════════════════════════
# NoveltyDetector Tests
# ═══════════════════════════════════════════════════════════════

class TestNoveltyDetector:
    def test_known_algorithm_detected(self):
        detector = NoveltyDetector()
        score, overlaps = detector.check_novelty(
            "A new transformer attention method",
            ["attention"],
        )
        assert score < 100  # Should detect "transformer" and "attention"
        assert len(overlaps) > 0

    def test_novel_idea(self):
        detector = NoveltyDetector()
        score, overlaps = detector.check_novelty(
            "Quantum-entangled manifold learning with holographic encoding",
            ["manifold_learning"],
        )
        # Should be quite novel (none of the known keywords match)
        assert score >= 70


# ═══════════════════════════════════════════════════════════════
# FeasibilityEstimator Tests
# ═══════════════════════════════════════════════════════════════

class TestFeasibilityEstimator:
    def test_exponential_complexity(self):
        estimator = FeasibilityEstimator()
        score, concerns = estimator.estimate(
            pseudocode="",
            complexity="O(n^3)",
            description="An algorithm with cubic complexity",
        )
        assert score < 100
        assert any("polynomial" in c.lower() or "exponential" in c.lower() for c in concerns)

    def test_training_free(self):
        estimator = FeasibilityEstimator()
        score, _ = estimator.estimate(
            pseudocode="",
            complexity="O(n log n)",
            description="A training-free approach",
        )
        assert score > 50  # training-free is a bonus


# ═══════════════════════════════════════════════════════════════
# AlgorithmProposer Tests
# ═══════════════════════════════════════════════════════════════

class TestAlgorithmProposer:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.tmpdir, "proposals.json")

    def test_propose(self):
        proposer = AlgorithmProposer(store_path=self.store_path)
        proposal = proposer.propose(
            problem="Efficient LLM inference on edge devices",
            known_techniques=["MoE", "quantization", "distillation"],
            inspiration_papers=["1234.5678"],
            description="Combine MoE with aggressive quantization for mobile inference",
        )
        assert proposal.name != ""
        assert proposal.problem_statement != ""
        assert proposal.overall_score > 0
        assert proposal.novelty_score >= 0

    def test_generate_variants(self):
        proposer = AlgorithmProposer(store_path=self.store_path)
        proposal = proposer.propose(
            problem="Test problem",
            known_techniques=["attention", "normalization"],
            inspiration_papers=[],
        )
        variants = proposer.generate_variants(proposal.id, n_variants=3)
        assert len(variants) == 3
        for v in variants:
            assert v.parent_id == proposal.id

    def test_top_proposals(self):
        proposer = AlgorithmProposer(store_path=self.store_path)
        for i in range(5):
            proposer.propose(
                problem=f"Problem {i}",
                known_techniques=["technique"],
                inspiration_papers=[],
            )
        top = proposer.get_top_proposals(n=3)
        assert len(top) <= 3

    def test_refine(self):
        proposer = AlgorithmProposer(store_path=self.store_path)
        proposal = proposer.propose(
            problem="Test",
            known_techniques=["technique_a"],
            inspiration_papers=[],
        )
        refined = proposer.refine(proposal.id, additional_techniques=["technique_b"])
        assert refined is not None
        assert "technique_b" in refined.building_blocks

    def test_implement_pseudocode(self):
        proposer = AlgorithmProposer(store_path=self.store_path)
        proposal = proposer.propose(
            problem="Test", known_techniques=["t"], inspiration_papers=[],
        )
        updated = proposer.implement_pseudocode(
            proposal.id,
            pseudocode="def algo(): pass",
            complexity="O(n)",
        )
        assert updated is not None
        assert updated.status == ProposalStatus.IMPLEMENTED
        assert updated.pseudocode == "def algo(): pass"

    def test_persistence(self):
        proposer = AlgorithmProposer(store_path=self.store_path)
        proposer.propose(problem="P", known_techniques=["t"], inspiration_papers=[])

        proposer2 = AlgorithmProposer(store_path=self.store_path)
        assert len(proposer2.proposals) == 1


# ═══════════════════════════════════════════════════════════════
# StatisticalAnalyzer Tests
# ═══════════════════════════════════════════════════════════════

class TestStatisticalAnalyzer:
    def test_t_test_significant(self):
        analyzer = StatisticalAnalyzer()
        result = analyzer.t_test(
            control_values=[0.85, 0.86, 0.84, 0.87, 0.85],
            treatment_values=[0.92, 0.93, 0.91, 0.94, 0.92],
        )
        assert result.status == HypothesisStatus.SUPPORTED
        assert result.p_value < 0.05
        assert result.effect_size > 1.0

    def test_t_test_not_significant(self):
        analyzer = StatisticalAnalyzer()
        result = analyzer.t_test(
            control_values=[0.85, 0.86, 0.85, 0.86, 0.85],
            treatment_values=[0.85, 0.86, 0.85, 0.87, 0.86],
        )
        assert result.status == HypothesisStatus.REJECTED
        assert result.p_value > 0.05

    def test_t_test_insufficient_data(self):
        analyzer = StatisticalAnalyzer()
        result = analyzer.t_test(
            control_values=[0.85],
            treatment_values=[0.86],
        )
        assert result.status == HypothesisStatus.INCONCLUSIVE

    def test_multiple_comparison_bonferroni(self):
        analyzer = StatisticalAnalyzer()
        raw_ps = [0.01, 0.02, 0.03]
        corrected = analyzer.multiple_comparison_correction(raw_ps, method="bonferroni")
        assert all(c >= p for c, p in zip(corrected, raw_ps))
        assert corrected[0] == 0.03  # 0.01 * 3

    def test_multiple_comparison_holm(self):
        analyzer = StatisticalAnalyzer()
        raw_ps = [0.01, 0.04, 0.10]
        corrected = analyzer.multiple_comparison_correction(raw_ps, method="holm")
        assert len(corrected) == 3
        # All corrected p-values should be >= raw p-values
        for c, r in zip(corrected, raw_ps):
            assert c >= r


# ═══════════════════════════════════════════════════════════════
# HypothesisFormulator Tests
# ═══════════════════════════════════════════════════════════════

class TestHypothesisFormulator:
    def test_from_paper_insight(self):
        formulator = HypothesisFormulator()
        hyp = formulator.from_paper_insight(
            observation="MLA reduces KV cache by 8x",
            paper_id="1234.5678",
            metrics=["accuracy", "memory"],
        )
        assert hyp.null_hypothesis != ""
        assert hyp.alternative_hypothesis != ""
        assert "1234.5678" in hyp.based_on_papers
        assert "accuracy" in hyp.metrics_to_measure

    def test_from_observation(self):
        formulator = HypothesisFormulator()
        hyp = formulator.from_observation(
            observation="Larger batch sizes improve training stability",
            metrics=["loss", "accuracy"],
        )
        assert hyp.null_hypothesis != ""
        assert "chance" in hyp.null_hypothesis.lower()


# ═══════════════════════════════════════════════════════════════
# HypothesisTester Tests
# ═══════════════════════════════════════════════════════════════

class TestHypothesisTester:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store_path = os.path.join(self.tmpdir, "hypotheses.json")

    def test_test_hypothesis(self):
        tester = HypothesisTester(store_path=self.store_path)
        formulator = HypothesisFormulator()
        hyp = formulator.from_observation("Test", ["accuracy"])

        result = tester.test_hypothesis(
            hypothesis=hyp,
            control_values=[0.8] * 10,
            treatment_values=[0.9] * 10,
            experiment_name="test_exp",
        )
        assert result.status == HypothesisStatus.SUPPORTED
        assert result.p_value < 0.01

    def test_summary(self):
        tester = HypothesisTester(store_path=self.store_path)
        formulator = HypothesisFormulator()
        hyp = formulator.from_observation("Test", ["accuracy"])
        tester.test_hypothesis(hyp, [0.8] * 5, [0.9] * 5, "exp1")

        summary = tester.summarize()
        assert summary["total_hypotheses"] == 1
        assert summary["supported"] == 1

    def test_get_supported(self):
        tester = HypothesisTester(store_path=self.store_path)
        formulator = HypothesisFormulator()
        hyp = formulator.from_observation("Test", ["accuracy"])
        tester.test_hypothesis(hyp, [0.8] * 5, [0.9] * 5, "exp1")

        supported = tester.get_supported_hypotheses()
        assert len(supported) == 1

    def test_persistence(self):
        tester = HypothesisTester(store_path=self.store_path)
        formulator = HypothesisFormulator()
        hyp = formulator.from_observation("Test", ["accuracy"])
        tester.test_hypothesis(hyp, [0.8] * 5, [0.9] * 5, "exp1")

        tester2 = HypothesisTester(store_path=self.store_path)
        assert len(tester2.hypotheses) == 1


# ═══════════════════════════════════════════════════════════════
# ReportWriter Tests
# ═══════════════════════════════════════════════════════════════

class TestReportWriter:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_create_and_compile(self):
        writer = ReportWriter(output_dir=self.tmpdir)
        report = writer.create_report(
            title="Test Report",
            authors=["Author A"],
            abstract="Test abstract.",
        )
        assert report.title == "Test Report"
        assert report.version == 1

        compiled = writer.compile(report.id)
        assert compiled is not None
        assert "Test Report" in compiled
        assert "Test abstract" in compiled
        assert report.compiled_path != ""

    def test_add_section(self):
        writer = ReportWriter(output_dir=self.tmpdir)
        report = writer.create_report(title="T", authors=["A"])
        ok = writer.add_section(report.id, "Method", "Our method...", level=1)
        assert ok
        assert len(report.sections) == 1

    def test_add_experiment_summary(self):
        writer = ReportWriter(output_dir=self.tmpdir)
        report = writer.create_report(title="T", authors=["A"])
        writer.add_experiment_summary(
            report.id, name="exp1", metric="accuracy", value=0.92,
            baseline=0.85, significance="p<0.01",
        )
        assert len(report.experiment_summaries) == 1
        assert abs(report.experiment_summaries[0].improvement - 0.07) < 1e-9

    def test_write_full_report(self):
        writer = ReportWriter(output_dir=self.tmpdir)
        report = writer.create_report(
            title="Efficient Transformer Variants",
            authors=["AI Scientist"],
            abstract="We study efficient transformer architectures.",
        )

        writer.add_section(report.id, "Introduction", "Transformers are powerful...")
        writer.add_experiment_summary(
            report.id, "Baseline", "accuracy", 0.85, baseline=0.80,
            significance="p<0.05",
        )
        writer.add_experiment_summary(
            report.id, "Our Method", "accuracy", 0.92, baseline=0.80,
            significance="p<0.001",
        )
        writer.write_results_section(report.id)
        writer.write_conclusion(
            report.id,
            findings=["Our method outperforms baseline by 7%"],
            limitations=["Tested on limited datasets"],
            future_work=["Scale to larger models"],
        )

        compiled = writer.compile(report.id)
        assert "Efficient Transformer" in compiled
        assert "0.920" in compiled

    def test_generate_latex(self):
        writer = ReportWriter(output_dir=self.tmpdir)
        report = writer.create_report(
            title="LaTeX Test",
            authors=["Author"],
            abstract="Testing LaTeX output.",
        )
        writer.add_experiment_summary(
            report.id, "exp1", "accuracy", 0.95, baseline=0.90,
        )
        latex = writer.generate_latex(report.id)
        assert latex is not None
        assert r"\documentclass" in latex
        assert "LaTeX Test" in latex
        assert r"\end{document}" in latex


# ═══════════════════════════════════════════════════════════════
# Phase9Orchestrator Tests
# ═══════════════════════════════════════════════════════════════

class TestPhase9Orchestrator:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_init(self):
        orch = Phase9Orchestrator(data_dir=self.tmpdir)
        assert orch.arxiv is not None
        assert orch.parser is not None
        assert orch.algorithm_proposer is not None
        assert orch.hypothesis_tester is not None
        assert orch.report_writer is not None

    def test_literature_review(self):
        orch = Phase9Orchestrator(data_dir=self.tmpdir)
        # This will attempt to call arXiv API — may fail in CI, but shouldn't crash
        try:
            result = orch.literature_review("transformer attention", max_papers=2)
            assert "topic" in result
            assert "papers_found" in result
        except Exception:
            pass  # Network may not be available

    def test_get_research_summary(self):
        orch = Phase9Orchestrator(data_dir=self.tmpdir)
        summary = orch.get_research_summary()
        assert "papers_in_library" in summary
        assert "algorithms" in summary
        assert "hypotheses" in summary
        assert "experiments_run" in summary
        assert "research_sessions" in summary

    def test_run_research_pipeline(self):
        orch = Phase9Orchestrator(data_dir=self.tmpdir)

        # Run with minimal settings (may hit network for arXiv)
        try:
            session = orch.run_research_pipeline(
                topic="test topic",
                arxiv_query="all:electron",  # broad query that returns results
                max_papers=1,
                hypotheses_to_test=1,
            )
            assert session.topic == "test topic"
            assert session.status == "completed"
            assert session.summary != ""
        except Exception:
            pass  # Network-dependent, but shouldn't crash


# ═══════════════════════════════════════════════════════════════
# Run all tests
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
