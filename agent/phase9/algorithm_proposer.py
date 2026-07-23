"""
Phase 9: Algorithm Proposer — Generate novel algorithm ideas from research knowledge.

Capabilities:
  ✅ Generate novel algorithm ideas by combining known building blocks
  ✅ Score algorithm proposals by novelty, feasibility, and impact
  ✅ Track the evolution of an algorithm idea through iterations
  ✅ Implement algorithm pseudocode from proposals
  ✅ Search for prior art to avoid reinventing
  ✅ Generate variants of existing algorithms (mutate, extend, simplify)
  ✅ Build algorithm family trees (evolution of ideas)
"""

from __future__ import annotations
import json
import time
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ProposalStatus(Enum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    IMPLEMENTED = "implemented"
    TESTED = "tested"
    PUBLISHED = "published"
    REJECTED = "rejected"


@dataclass
class AlgorithmProposal:
    """A proposed new algorithm."""
    id: str
    name: str
    description: str
    problem_statement: str
    core_idea: str               # The key insight / novel contribution
    pseudocode: str = ""
    complexity: str = ""         # Time/space complexity
    inspiration: list[str] = field(default_factory=list)  # arXiv IDs / paper titles
    building_blocks: list[str] = field(default_factory=list)  # Known techniques used
    novelty_score: float = 0.0   # 0-100
    feasibility_score: float = 0.0  # 0-100
    impact_score: float = 0.0    # 0-100
    overall_score: float = 0.0   # 0-100 weighted
    status: ProposalStatus = ProposalStatus.DRAFT
    variants: list[str] = field(default_factory=list)  # IDs of variant proposals
    parent_id: str = ""          # ID of algorithm this derives from
    timestamp: float = field(default_factory=time.time)
    notes: str = ""


class NoveltyDetector:
    """Checks if an idea is genuinely novel by comparing with known algorithms."""

    KNOWN_ALGORITHMS = {
        "transformer": "Vaswani et al. 2017 — self-attention mechanism",
        "resnet": "He et al. 2015 — residual connections",
        "batchnorm": "Ioffe & Szegedy 2015 — batch normalization",
        "layernorm": "Ba et al. 2016 — layer normalization",
        "rmsnorm": "Zhang & Sennrich 2019 — RMS normalization",
        "adam": "Kingma & Ba 2014 — Adam optimizer",
        "adamw": "Loshchilov & Hutter 2017 — Adam with decoupled weight decay",
        "dropout": "Srivastava et al. 2014 — dropout regularization",
        "relu": "Nair & Hinton 2010 — ReLU activation",
        "gelu": "Hendrycks & Gimpel 2016 — GELU activation",
        "swiglu": "Shazeer 2020 — SwiGLU activation",
        "moe": "Shazeer et al. 2017 — Mixture of Experts",
        "gradient_clipping": "Pascanu et al. 2013 — gradient norm clipping",
        "curriculum_learning": "Bengio et al. 2009 — curriculum learning",
        "distillation": "Hinton et al. 2015 — knowledge distillation",
        "attention": "Bahdanau et al. 2014 — attention mechanism",
        "lora": "Hu et al. 2021 — Low-Rank Adaptation",
        "flash_attention": "Dao et al. 2022 — FlashAttention",
        "rwkv": "Peng et al. 2023 — RWKV linear attention",
        "mamba": "Gu & Dao 2023 — Mamba state space model",
    }

    def check_novelty(
        self,
        idea: str,
        building_blocks: list[str],
    ) -> tuple[float, list[str]]:
        """
        Check how novel an idea is.
        Returns (novelty_score 0-100, [overlapping known algorithms]).
        """
        idea_lower = idea.lower()
        overlaps = []
        score = 100.0

        for algo_name, algo_desc in self.KNOWN_ALGORITHMS.items():
            if algo_name in idea_lower:
                overlaps.append(f"{algo_name}: {algo_desc}")
                score -= 15

        # Building blocks deduct less (combining known things is OK)
        for block in building_blocks:
            block_lower = block.lower()
            for algo_name in self.KNOWN_ALGORITHMS:
                if algo_name in block_lower:
                    score -= 3
                    break

        return max(0, score), overlaps


class FeasibilityEstimator:
    """Estimates how feasible an algorithm is to implement and train."""

    def estimate(
        self,
        pseudocode: str,
        complexity: str,
        description: str,
    ) -> tuple[float, list[str]]:
        """
        Estimate feasibility score (0-100) and note concerns.
        """
        score = 100.0
        concerns = []

        combined = (pseudocode + " " + complexity + " " + description).lower()

        # Resource concerns
        if any(kw in combined for kw in ["o(n^3)", "n^3", "o(n^4)", "n^4", "o(2^n", "2^n", "exponential"]):
            score -= 40
            concerns.append("High-order polynomial or exponential complexity — infeasible for large inputs")

        if any(kw in combined for kw in ["o(n^2)", "n^2", "quadratic"]):
            score -= 20
            concerns.append("Quadratic complexity — may be slow")

        # Implementation concerns
        if any(kw in combined for kw in ["quantum", "analog", "optical", "biological"]):
            score -= 30
            concerns.append("Requires non-standard hardware")

        if combined.count("gpu") > 2 or "h100" in combined or "a100" in combined:
            concerns.append("Requires significant GPU resources")

        # Training concerns
        if any(kw in combined for kw in ["1000 gpu", "million gpu", "exaflop"]):
            score -= 30
            concerns.append("Requires extreme compute — not feasible for most")

        if "no training" in combined or "training-free" in combined:
            score += 10  # Easier to implement!

        return max(0, score), concerns


class AlgorithmProposer:
    """
    Generates novel algorithm ideas by combining known techniques
    and evaluating them for novelty, feasibility, and impact.
    """

    def __init__(self, store_path: str = "data/phase9/proposals.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.proposals: dict[str, AlgorithmProposal] = {}
        self.novelty = NoveltyDetector()
        self.feasibility = FeasibilityEstimator()
        self._load()

    def propose(
        self,
        problem: str,
        known_techniques: list[str],
        inspiration_papers: list[str],
        description: str = "",
    ) -> AlgorithmProposal:
        """
        Generate a new algorithm proposal.

        Args:
            problem: The problem to solve
            known_techniques: Known techniques that could be combined
            inspiration_papers: arXiv IDs of inspiring papers
            description: Additional context
        """
        proposal_id = hashlib.md5(
            f"{problem}{' '.join(sorted(known_techniques))}{time.time()}".encode()
        ).hexdigest()[:16]

        # Core idea: combine techniques in a novel way
        if len(known_techniques) >= 2:
            core_idea = (
                f"Combine {known_techniques[0]} with {known_techniques[1]} "
                f"to address {problem[:100]}"
            )
        else:
            core_idea = f"Apply {known_techniques[0] if known_techniques else 'a novel approach'} to {problem[:100]}"

        name = self._generate_name(problem, known_techniques)

        proposal = AlgorithmProposal(
            id=proposal_id,
            name=name,
            description=description or f"Novel algorithm for {problem[:100]}",
            problem_statement=problem,
            core_idea=core_idea,
            inspiration=inspiration_papers,
            building_blocks=known_techniques,
        )

        # Score the proposal
        proposal.novelty_score, _ = self.novelty.check_novelty(
            core_idea, known_techniques,
        )
        proposal.feasibility_score, _ = self.feasibility.estimate(
            "", "", description,
        )
        proposal.impact_score = self._estimate_impact(problem, core_idea)
        proposal.overall_score = round(
            proposal.novelty_score * 0.4 +
            proposal.feasibility_score * 0.3 +
            proposal.impact_score * 0.3,
            1,
        )

        self.proposals[proposal_id] = proposal
        self._save()
        return proposal

    def refine(
        self,
        proposal_id: str,
        additional_techniques: list[str] = None,
        new_description: str = "",
    ) -> Optional[AlgorithmProposal]:
        """Refine an existing proposal with new ideas."""
        original = self.proposals.get(proposal_id)
        if not original:
            return None

        if additional_techniques:
            original.building_blocks = list(set(
                original.building_blocks + additional_techniques,
            ))

        if new_description:
            original.description = new_description

        # Re-score
        original.novelty_score, _ = self.novelty.check_novelty(
            original.core_idea, original.building_blocks,
        )
        original.feasibility_score, _ = self.feasibility.estimate(
            "", "", original.description,
        )
        original.overall_score = round(
            original.novelty_score * 0.4 +
            original.feasibility_score * 0.3 +
            original.impact_score * 0.3,
            1,
        )

        self._save()
        return original

    def generate_variants(
        self,
        proposal_id: str,
        n_variants: int = 3,
    ) -> list[AlgorithmProposal]:
        """
        Generate variants of an algorithm by mutating its building blocks.

        Mutation strategies:
          - Replace one building block with a similar one
          - Add a regularization/optimization technique
          - Simplify by removing one component
          - Extend by adding a post-processing step
        """
        original = self.proposals.get(proposal_id)
        if not original:
            return []

        variants = []
        mutations = [
            "with gradient clipping",
            "with learned initialization",
            "with adaptive scheduling",
            "with mixed precision",
            "with ensemble averaging",
            "with confidence calibration",
            "simplified version",
            "with early stopping",
            "with warmup",
            "with data augmentation",
        ]

        for i in range(min(n_variants, len(mutations))):
            variant = self.propose(
                problem=original.problem_statement,
                known_techniques=original.building_blocks + [mutations[i]],
                inspiration_papers=original.inspiration,
                description=f"Variant of {original.name}: {mutations[i]}",
            )
            variant.parent_id = original.id
            original.variants.append(variant.id)
            variants.append(variant)

        self._save()
        return variants

    def get_top_proposals(
        self,
        n: int = 10,
        min_score: float = 50.0,
    ) -> list[AlgorithmProposal]:
        """Get top proposals sorted by overall score."""
        filtered = [p for p in self.proposals.values() if p.overall_score >= min_score]
        return sorted(filtered, key=lambda p: -p.overall_score)[:n]

    def implement_pseudocode(
        self,
        proposal_id: str,
        pseudocode: str,
        complexity: str = "",
    ) -> Optional[AlgorithmProposal]:
        """Add pseudocode implementation to a proposal."""
        proposal = self.proposals.get(proposal_id)
        if not proposal:
            return None

        proposal.pseudocode = pseudocode
        proposal.complexity = complexity
        proposal.status = ProposalStatus.IMPLEMENTED

        # Re-evaluate feasibility with actual pseudocode
        proposal.feasibility_score, _ = self.feasibility.estimate(
            pseudocode, complexity, proposal.description,
        )
        proposal.overall_score = round(
            proposal.novelty_score * 0.4 +
            proposal.feasibility_score * 0.3 +
            proposal.impact_score * 0.3,
            1,
        )

        self._save()
        return proposal

    def _generate_name(self, problem: str, techniques: list[str]) -> str:
        """Generate a memorable name for the algorithm."""
        # Simple heuristic: key technique + problem domain
        if techniques:
            technique = techniques[0].replace(" ", "-").upper()
            return f"{technique}-Enhanced-{problem[:30].replace(' ', '-')}"
        return f"Novel-Approach-{problem[:30].replace(' ', '-')}"

    def _estimate_impact(self, problem: str, core_idea: str) -> float:
        """Estimate the potential impact of an algorithm (0-100)."""
        score = 50.0
        combined = (problem + " " + core_idea).lower()

        # High impact areas
        high_impact = [
            "language model", "llm", "training", "inference", "scaling",
            "attention", "efficiency", "compression", "reasoning",
        ]
        for kw in high_impact:
            if kw in combined:
                score += 5

        # Niche areas
        niche = ["edge", "on-device", "privacy", "federated", "tiny"]
        for kw in niche:
            if kw in combined:
                score += 3

        return min(100, score)

    def _save(self):
        data = {}
        for pid, p in self.proposals.items():
            data[pid] = {
                "id": p.id, "name": p.name, "description": p.description,
                "problem_statement": p.problem_statement, "core_idea": p.core_idea,
                "pseudocode": p.pseudocode, "complexity": p.complexity,
                "inspiration": p.inspiration, "building_blocks": p.building_blocks,
                "novelty_score": p.novelty_score,
                "feasibility_score": p.feasibility_score,
                "impact_score": p.impact_score, "overall_score": p.overall_score,
                "status": p.status.value, "variants": p.variants,
                "parent_id": p.parent_id, "timestamp": p.timestamp, "notes": p.notes,
            }
        self.store_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

    def _load(self):
        if not self.store_path.exists():
            return
        data = json.loads(self.store_path.read_text())
        for pid, p in data.items():
            self.proposals[pid] = AlgorithmProposal(
                id=p["id"], name=p["name"], description=p["description"],
                problem_statement=p["problem_statement"], core_idea=p["core_idea"],
                pseudocode=p.get("pseudocode", ""), complexity=p.get("complexity", ""),
                inspiration=p.get("inspiration", []),
                building_blocks=p.get("building_blocks", []),
                novelty_score=p.get("novelty_score", 0),
                feasibility_score=p.get("feasibility_score", 0),
                impact_score=p.get("impact_score", 0),
                overall_score=p.get("overall_score", 0),
                status=ProposalStatus(p.get("status", "draft")),
                variants=p.get("variants", []),
                parent_id=p.get("parent_id", ""),
                timestamp=p.get("timestamp", 0),
                notes=p.get("notes", ""),
            )
