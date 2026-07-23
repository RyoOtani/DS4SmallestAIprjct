"""Training Methods — MPS, LoRA, DPO, Distillation, Instruction Tuning."""
from .mps_trainer import MPSTrainer, MPSConfig, MPSDetector
from .methods import (
    LoRATrainer, QLoRATrainer, DPOTrainer,
    DistillationTrainer, InstructionTuner,
    TrainingPreset, TrainingResult, TrainingMethodRegistry,
    TRAINING_PRESETS,
)

__all__ = [
    # MPS
    "MPSTrainer", "MPSConfig", "MPSDetector",
    # Training methods
    "LoRATrainer", "QLoRATrainer", "DPOTrainer",
    "DistillationTrainer", "InstructionTuner",
    "TrainingPreset", "TrainingResult", "TrainingMethodRegistry",
    "TRAINING_PRESETS",
]
