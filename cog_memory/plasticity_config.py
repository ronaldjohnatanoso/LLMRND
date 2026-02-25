"""Configuration for neuroplasticity and connection weight learning."""

from dataclasses import dataclass


@dataclass
class PlasticityConfig:
    """Configuration for neuroplasticity and connection weight learning.

    Attributes:
        learning_rate: How much to strengthen connections per use (0-1)
        learning_mode: Type of learning ("hebbian", "reinforcement", "consolidation")
        min_weight: Minimum connection weight (prevents deletion)
        max_weight: Maximum connection weight (prevents explosion)
        decay_enabled: Whether to decay unused connections
        decay_rate: Per-query decay for unused connections
        decay_threshold: Prune connections below this weight during consolidation
        consolidation_enabled: Whether to run periodic consolidation
        consolidation_interval: Run consolidation every N queries
        consolidation_strength_threshold: Usage count to strengthen during consolidation
        consolidation_prune_threshold: Usage count to prune during consolidation
        reinforcement_enabled: Whether to accept user feedback
        positive_feedback_boost: Weight increase for positive feedback
        negative_feedback_penalty: Weight decrease for negative feedback
    """

    # Hebbian learning parameters
    learning_rate: float = 0.02
    learning_mode: str = "hebbian"  # "hebbian", "reinforcement", "consolidation"

    # Weight constraints
    min_weight: float = 0.05
    max_weight: float = 1.0

    # Decay (forgetting)
    decay_enabled: bool = False
    decay_rate: float = 0.001
    decay_threshold: float = 0.1

    # Consolidation (periodic strengthening/pruning)
    consolidation_enabled: bool = True
    consolidation_interval: int = 100
    consolidation_strength_threshold: int = 10
    consolidation_prune_threshold: int = 2

    # Reinforcement learning (user feedback)
    reinforcement_enabled: bool = False
    positive_feedback_boost: float = 0.1
    negative_feedback_penalty: float = 0.2
