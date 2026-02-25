"""Configuration for tiered propagation limits based on signal quality.

Implements adaptive fan-out limits where nodes with stronger connections
can propagate to more neighbors, mimicking cognitive attention mechanisms.
"""

from dataclasses import dataclass, field
from cog_memory.node import Role


@dataclass
class PropagationConfig:
    """Configuration for tiered propagation limits based on signal quality.

    The system uses quality tiers to determine how many neighbors each node
    can propagate to. Nodes with stronger connections get higher limits,
    while nodes with weaker connections are more conservative.

    Quality score is calculated as:
        score = (max_weight * peak_weight) + (avg_weight * avg_weight)

    Attributes:
        tiers: Dictionary mapping tier names to (threshold, limit) tuples
        role_modifiers: Multipliers for different node roles
        peak_weight: Weight for max connection in quality score (0-1)
        avg_weight: Weight for avg connection in quality score (0-1)
        hard_cap_multiplier: Max multiplier of tier limit (prevents explosion)
        hop_decay_enabled: Whether to reduce limits for deeper hops
        hop_decay_factor: How much to reduce per hop (0-1)
    """

    # Tier thresholds (quality_score -> max_neighbors)
    # IMPORTANT: Must be ordered from highest to lowest threshold!
    tiers: dict[str, tuple[float, int]] = field(default_factory=lambda: {
        "excellent": (0.90, 15),  # 90%+ strength → 15 neighbors
        "good": (0.75, 10),       # 75%+ strength → 10 neighbors
        "moderate": (0.60, 7),    # 60%+ strength → 7 neighbors
        "weak": (0.45, 5),        # 45%+ strength → 5 neighbors
        "poor": (0.30, 3),        # 30%+ strength → 3 neighbors
        "noise": (0.0, 1),        # Below 30% → 1 neighbor
    })

    # Role modifiers (multipliers for base limits)
    role_modifiers: dict[Role, float] = field(default_factory=lambda: {
        Role.GOAL: 1.5,
        Role.DECISION: 1.2,
        Role.CONDITIONAL_DEPENDENCY: 1.0,
        Role.CONSTRAINT: 1.0,
        Role.FACT: 1.0,
        Role.OBSERVATION: 0.8,
    })

    # Quality score calculation (70% peak, 30% average)
    peak_weight: float = 0.7
    avg_weight: float = 0.3

    # Hard cap to prevent explosion (multiplier of tier limit)
    hard_cap_multiplier: float = 2.0

    # Hop-based decay (reduce limits for deeper hops)
    hop_decay_enabled: bool = True
    hop_decay_factor: float = 0.5  # Each hop reduces limit by 50%

    def get_limit(self, node, hop: int) -> int:
        """Calculate propagation limit for a node at given hop.

        Args:
            node: The Node object to calculate limit for
            hop: Current hop level (0-indexed)

        Returns:
            Maximum number of neighbors this node can propagate to
        """
        weights = list(node.neighbors.values())
        if not weights:
            return 0

        max_weight = max(weights)
        avg_weight = sum(weights) / len(weights)

        # Quality score: hybrid of peak and average
        quality_score = (max_weight * self.peak_weight) + (avg_weight * self.avg_weight)

        # Find appropriate tier (highest threshold that matches)
        base_limit = 1
        for tier_name, (threshold, limit) in self.tiers.items():
            if quality_score >= threshold:
                base_limit = limit
                break  # Found the highest matching tier
        # If no tier matched (shouldn't happen), base_limit stays at 1

        # Apply role modifier
        role_mod = self.role_modifiers.get(node.role, 1.0)
        limit = int(base_limit * role_mod)

        # Apply hop decay
        if self.hop_decay_enabled and hop > 0:
            decay = self.hop_decay_factor ** hop
            limit = max(1, int(limit * decay))

        # Apply hard cap
        hard_cap = int(base_limit * self.hard_cap_multiplier)

        return max(1, min(limit, hard_cap))
