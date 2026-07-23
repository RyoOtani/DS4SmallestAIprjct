"""
Phase 8: Autonomous Self-Improvement System.

This module implements a closed-loop AI that can:
  - Self-evaluate its own code and actions
  - Run improvement cycles (diagnose → fix → measure)
  - Learn from past experiences via replay
  - Update LoRA adapters online without forgetting
  - Learn meta-patterns across different tasks
  - Coordinate all of the above through a unified orchestrator
"""

from .self_improve import (
    SelfEvaluator,
    ImprovementCycle,
    ImprovementRecord,
    ImprovementStatus,
)

from .memory_evolution import (
    Experience,
    ExperienceReplay,
    FailureDatabase,
    FailurePattern,
    KnowledgeCompressor,
)

from .online_learning import (
    OnlineLearner,
    LoRASnapshot,
)

from .meta_learning import (
    MetaLearner,
    SkillTemplate,
    TaskProfile,
)

from .orchestrator import (
    Phase8Orchestrator,
    SelfImprovementReport,
)

__all__ = [
    # Self-improvement core
    "SelfEvaluator",
    "ImprovementCycle",
    "ImprovementRecord",
    "ImprovementStatus",
    # Memory & experience
    "Experience",
    "ExperienceReplay",
    "FailureDatabase",
    "FailurePattern",
    "KnowledgeCompressor",
    # Online learning
    "OnlineLearner",
    "LoRASnapshot",
    # Meta learning
    "MetaLearner",
    "SkillTemplate",
    "TaskProfile",
    # Orchestrator
    "Phase8Orchestrator",
    "SelfImprovementReport",
]

