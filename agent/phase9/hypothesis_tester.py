"""
Phase 9: Hypothesis Tester — Form and test research hypotheses scientifically.

Capabilities:
  ✅ Form testable hypotheses from observations and paper insights
  ✅ Design controlled experiments to test each hypothesis
  ✅ Statistical analysis of results (p-value, effect size, confidence intervals)
  ✅ Accept/reject hypotheses with confidence levels
  ✅ Track hypothesis testing history
  ✅ Generate hypothesis → experiment → result chains
  ✅ Avoid common statistical pitfalls (p-hacking, multiple comparisons)
"""

from __future__ import annotations
import json
import time
import math
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class HypothesisStatus(Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"
    REFINED = "refined"  # Modified based on results


class ConfidenceLevel(Enum):
    HIGH = "high"        # p < 0.001
    MEDIUM = "medium"    # p < 0.01
    LOW = "low"          # p < 0.05
    NONE = "none"        # p >= 0.05


@dataclass
class Hypothesis:
    """A scientific hypothesis to test."""
    id: str
    statement: str                # The hypothesis (null + alternative)
    null_hypothesis: str          # H₀: no effect
    alternative_hypothesis: str   # H₁: there is an effect
    motivation: str = ""          # Why this hypothesis?
    based_on_papers: list[str] = field(default_factory=list)  # arXiv IDs
    predicted_effect: str = ""    # Expected outcome
    metrics_to_measure: list[str] = field(default_factory=list)
    experiment_design: dict = field(default_factory=dict)
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: ConfidenceLevel = ConfidenceLevel.NONE
    p_value: float = 1.0
    effect_size: float = 0.0
    sample_size: int = 0
    result_summary: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class HypothesisTestResult:
    """Results from testing a single hypothesis."""
    hypothesis_id: str
    experiment_name: str
    status: HypothesisStatus
    p_value: float
    effect_size: float
    confidence_interval: tuple[float, float]
    sample_size: int
    control_mean: float
    treatment_mean: float
    delta: float
    conclusion: str
    timestamp: float = field(default_factory=time.time)


class HypothesisFormulator:
    """Formulates testable hypotheses from papers and observations."""

    def from_paper_insight(
        self,
        observation: str,
        paper_id: str,
        metrics: list[str],
    ) -> Hypothesis:
        """
        Form a hypothesis from a paper's findings.

        Example:
          observation = "Paper X shows MLA reduces KV cache by 8x without quality loss"
          → H₀: MLA does not affect generation quality
          → H₁: MLA reduces KV cache while maintaining quality within 1% of full attention
        """
        hyp_id = hashlib.md5(
            f"{observation}{paper_id}{time.time()}".encode()
        ).hexdigest()[:16]

        return Hypothesis(
            id=hyp_id,
            statement=observation,
            null_hypothesis=f"No significant effect: {observation[:150]}",
            alternative_hypothesis=f"Significant positive effect: {observation[:150]}",
            motivation=f"Inspired by paper {paper_id}",
            based_on_papers=[paper_id],
            metrics_to_measure=metrics,
        )

    def from_algorithm_proposal(
        self,
        proposal_name: str,
        proposal_id: str,
        expected_improvement: str,
        baseline_metric: str,
        metrics: list[str],
    ) -> Hypothesis:
        """Form a hypothesis to test a proposed algorithm."""
        return Hypothesis(
            id=hashlib.md5(
                f"{proposal_name}{proposal_id}{time.time()}".encode()
            ).hexdigest()[:16],
            statement=f"Algorithm '{proposal_name}' {expected_improvement}",
            null_hypothesis=f"'{proposal_name}' does not improve {baseline_metric}",
            alternative_hypothesis=f"'{proposal_name}' improves {baseline_metric} by >1%",
            motivation=f"Test proposed algorithm: {proposal_id}",
            based_on_papers=[],
            metrics_to_measure=metrics,
            predicted_effect=expected_improvement,
        )

    def from_observation(
        self,
        observation: str,
        metrics: list[str],
    ) -> Hypothesis:
        """Form a hypothesis from an empirical observation."""
        hyp_id = hashlib.md5(
            f"{observation}{time.time()}".encode()
        ).hexdigest()[:16]

        return Hypothesis(
            id=hyp_id,
            statement=f"Observed: {observation}",
            null_hypothesis=f"The observation '{observation[:100]}' is due to chance",
            alternative_hypothesis=f"The observation '{observation[:100]}' reflects a real effect",
            motivation="Empirical observation during experiments",
            metrics_to_measure=metrics,
        )


class StatisticalAnalyzer:
    """Performs statistical analysis on experiment results."""

    def t_test(
        self,
        control_values: list[float],
        treatment_values: list[float],
        alpha: float = 0.05,
    ) -> HypothesisTestResult:
        """
        Welch's t-test (unequal variance).

        Returns (p_value, effect_size_cohens_d, confidence_interval).
        """
        n1, n2 = len(control_values), len(treatment_values)
        if n1 < 2 or n2 < 2:
            return HypothesisTestResult(
                hypothesis_id="", experiment_name="",
                status=HypothesisStatus.INCONCLUSIVE,
                p_value=1.0, effect_size=0.0,
                confidence_interval=(0, 0), sample_size=n1 + n2,
                control_mean=0, treatment_mean=0, delta=0,
                conclusion="Insufficient data",
            )

        mean1 = sum(control_values) / n1
        mean2 = sum(treatment_values) / n2

        var1 = sum((x - mean1) ** 2 for x in control_values) / (n1 - 1) if n1 > 1 else 0
        var2 = sum((x - mean2) ** 2 for x in treatment_values) / (n2 - 1) if n2 > 1 else 0

        # Welch's t-statistic
        se = math.sqrt(var1 / n1 + var2 / n2)
        if se == 0:
            t_stat = float("inf") if mean2 != mean1 else 0.0
        else:
            t_stat = (mean2 - mean1) / se

        # Degrees of freedom (Welch-Satterthwaite)
        if var1 == 0 and var2 == 0:
            df = n1 + n2 - 2
        else:
            num = (var1 / n1 + var2 / n2) ** 2
            denom = ((var1 / n1) ** 2 / (n1 - 1)) + ((var2 / n2) ** 2 / (n2 - 1))
            df = num / denom if denom != 0 else n1 + n2 - 2

        # Approximate p-value from t-distribution
        p_value = self._t_cdf_approx(abs(t_stat), df) * 2
        p_value = min(p_value, 1.0)

        # Cohen's d (effect size)
        pooled_std = math.sqrt((var1 * (n1 - 1) + var2 * (n2 - 1)) / (n1 + n2 - 2)) if (n1 + n2) > 2 else 1.0
        effect_size = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0.0

        # 95% confidence interval
        ci_margin = 1.96 * se  # approx for df>=30
        ci = (mean2 - mean1 - ci_margin, mean2 - mean1 + ci_margin)

        # Conclusion
        if p_value < 0.001:
            status = HypothesisStatus.SUPPORTED
            conclusion = f"Strongly supported (p={p_value:.4f}, d={effect_size:.2f})"
        elif p_value < alpha:
            status = HypothesisStatus.SUPPORTED
            conclusion = f"Supported (p={p_value:.4f}, d={effect_size:.2f})"
        else:
            status = HypothesisStatus.REJECTED
            conclusion = f"Not supported (p={p_value:.4f}, insufficient evidence)"

        return HypothesisTestResult(
            hypothesis_id="", experiment_name="",
            status=status,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            sample_size=n1 + n2,
            control_mean=mean1,
            treatment_mean=mean2,
            delta=mean2 - mean1,
            conclusion=conclusion,
        )

    def _t_cdf_approx(self, t: float, df: float) -> float:
        """Approximate two-tailed p-value from t-distribution."""
        if df <= 0:
            return 1.0
        # Using the Abramowitz & Stegun approximation
        x = df / (df + t * t)
        # Regularized incomplete beta function approximation
        if df <= 1:
            return 1.0 - 2 * math.atan(abs(t)) / math.pi
        # Simplified approximation for df > 1
        if abs(t) > 10:
            return 0.0
        # Approximation using normal distribution for large df
        if df > 30:
            # Wilson-Hilferty approximation
            z = abs(t)
            return 2 * (1 - self._norm_cdf_approx(z))
        # Fallback: rough estimate
        return min(1.0, 1.0 / (1.0 + t * t / df) ** ((df + 1) / 2))

    def _norm_cdf_approx(self, z: float) -> float:
        """Approximate normal CDF."""
        # Abramowitz & Stegun 26.2.17
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    def multiple_comparison_correction(
        self,
        p_values: list[float],
        method: str = "bonferroni",
    ) -> list[float]:
        """
        Apply correction for multiple hypothesis testing.

        Args:
            p_values: Raw p-values
            method: 'bonferroni' or 'holm'
        """
        if method == "bonferroni":
            n = len(p_values)
            return [min(p * n, 1.0) for p in p_values]

        # Holm-Bonferroni (step-down)
        n = len(p_values)
        indexed = sorted(enumerate(p_values), key=lambda x: x[1])
        corrected = [0.0] * n
        for rank, (idx, p) in enumerate(indexed):
            corrected[idx] = min(p * (n - rank), 1.0)
            # Ensure monotonicity
            if rank > 0:
                prev_idx = indexed[rank - 1][0]
                corrected[idx] = max(corrected[idx], corrected[prev_idx])
        return corrected


class HypothesisTester:
    """
    End-to-end hypothesis testing system.

    Flow:
      1. Form hypothesis from observation/paper/proposal
      2. Design experiment to test it
      3. Run experiment
      4. Analyze results statistically
      5. Accept/reject/refine hypothesis
    """

    def __init__(self, store_path: str = "data/phase9/hypotheses.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.hypotheses: dict[str, Hypothesis] = {}
        self.results: list[HypothesisTestResult] = []
        self.formulator = HypothesisFormulator()
        self.analyzer = StatisticalAnalyzer()
        self._load()

    def test_hypothesis(
        self,
        hypothesis: Hypothesis,
        control_values: list[float],
        treatment_values: list[float],
        experiment_name: str = "",
        alpha: float = 0.05,
    ) -> HypothesisTestResult:
        """Test a hypothesis with actual data."""
        result = self.analyzer.t_test(control_values, treatment_values, alpha)
        result.hypothesis_id = hypothesis.id
        result.experiment_name = experiment_name

        hypothesis.status = result.status
        hypothesis.p_value = result.p_value
        hypothesis.effect_size = result.effect_size
        hypothesis.sample_size = result.sample_size
        hypothesis.result_summary = result.conclusion

        if result.p_value < 0.001:
            hypothesis.confidence = ConfidenceLevel.HIGH
        elif result.p_value < 0.01:
            hypothesis.confidence = ConfidenceLevel.MEDIUM
        elif result.p_value < alpha:
            hypothesis.confidence = ConfidenceLevel.LOW
        else:
            hypothesis.confidence = ConfidenceLevel.NONE

        self.hypotheses[hypothesis.id] = hypothesis
        self.results.append(result)
        self._save()
        return result

    def test_batch(
        self,
        hypotheses_data: list[dict],
        alpha: float = 0.05,
        correct_multiple: bool = True,
    ) -> list[HypothesisTestResult]:
        """
        Test multiple hypotheses and apply multiple comparison correction.

        Args:
            hypotheses_data: List of {hypothesis, control_values, treatment_values}
            alpha: Significance level
            correct_multiple: Apply Bonferroni/Holm correction
        """
        results = []
        raw_p_values = []

        for hd in hypotheses_data:
            result = self.test_hypothesis(
                hypothesis=hd["hypothesis"],
                control_values=hd["control_values"],
                treatment_values=hd["treatment_values"],
                experiment_name=hd.get("experiment_name", ""),
                alpha=alpha,
            )
            results.append(result)
            raw_p_values.append(result.p_value)

        # Apply multiple comparison correction
        if correct_multiple and len(results) > 1:
            corrected_ps = self.analyzer.multiple_comparison_correction(
                raw_p_values, method="holm",
            )
            for i, result in enumerate(results):
                result.p_value = corrected_ps[i]
                if corrected_ps[i] >= alpha and result.status == HypothesisStatus.SUPPORTED:
                    result.status = HypothesisStatus.REJECTED
                    result.conclusion += " (not significant after multiple comparison correction)"

        return results

    def get_supported_hypotheses(self) -> list[Hypothesis]:
        """Get all supported hypotheses."""
        return [
            h for h in self.hypotheses.values()
            if h.status == HypothesisStatus.SUPPORTED
        ]

    def summarize(self) -> dict:
        """Get summary statistics of hypothesis testing."""
        total = len(self.hypotheses)
        supported = sum(
            1 for h in self.hypotheses.values()
            if h.status == HypothesisStatus.SUPPORTED
        )
        rejected = sum(
            1 for h in self.hypotheses.values()
            if h.status == HypothesisStatus.REJECTED
        )

        return {
            "total_hypotheses": total,
            "supported": supported,
            "rejected": rejected,
            "inconclusive": total - supported - rejected,
            "support_rate": f"{supported / total:.0%}" if total > 0 else "N/A",
            "avg_effect_size": (
                sum(h.effect_size for h in self.hypotheses.values()) / total
                if total > 0 else 0
            ),
        }

    def _save(self):
        data = {
            "hypotheses": {
                hid: {
                    "id": h.id, "statement": h.statement,
                    "null_hypothesis": h.null_hypothesis,
                    "alternative_hypothesis": h.alternative_hypothesis,
                    "motivation": h.motivation,
                    "based_on_papers": h.based_on_papers,
                    "predicted_effect": h.predicted_effect,
                    "metrics_to_measure": h.metrics_to_measure,
                    "status": h.status.value,
                    "confidence": h.confidence.value,
                    "p_value": h.p_value, "effect_size": h.effect_size,
                    "sample_size": h.sample_size,
                    "result_summary": h.result_summary,
                    "timestamp": h.timestamp,
                }
                for hid, h in self.hypotheses.items()
            },
            "results": [
                {
                    "hypothesis_id": r.hypothesis_id,
                    "experiment_name": r.experiment_name,
                    "status": r.status.value,
                    "p_value": r.p_value, "effect_size": r.effect_size,
                    "sample_size": r.sample_size,
                    "control_mean": r.control_mean,
                    "treatment_mean": r.treatment_mean,
                    "delta": r.delta, "conclusion": r.conclusion,
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ],
        }
        self.store_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _load(self):
        if not self.store_path.exists():
            return
        data = json.loads(self.store_path.read_text())
        for hid, h in data.get("hypotheses", {}).items():
            self.hypotheses[hid] = Hypothesis(
                id=h["id"], statement=h["statement"],
                null_hypothesis=h["null_hypothesis"],
                alternative_hypothesis=h["alternative_hypothesis"],
                motivation=h.get("motivation", ""),
                based_on_papers=h.get("based_on_papers", []),
                predicted_effect=h.get("predicted_effect", ""),
                metrics_to_measure=h.get("metrics_to_measure", []),
                status=HypothesisStatus(h.get("status", "proposed")),
                confidence=ConfidenceLevel(h.get("confidence", "none")),
                p_value=h.get("p_value", 1.0),
                effect_size=h.get("effect_size", 0.0),
                sample_size=h.get("sample_size", 0),
                result_summary=h.get("result_summary", ""),
                timestamp=h.get("timestamp", 0),
            )
        for r in data.get("results", []):
            self.results.append(HypothesisTestResult(
                hypothesis_id=r["hypothesis_id"],
                experiment_name=r["experiment_name"],
                status=HypothesisStatus(r["status"]),
                p_value=r["p_value"], effect_size=r["effect_size"],
                confidence_interval=(0, 0),
                sample_size=r["sample_size"],
                control_mean=r["control_mean"],
                treatment_mean=r["treatment_mean"],
                delta=r["delta"], conclusion=r["conclusion"],
                timestamp=r.get("timestamp", 0),
            ))
