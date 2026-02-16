"""Core node representation for the cognitive graph.

A Node represents a commitment extracted from text with a specific meta-role.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """Meta-roles for commitments in the cognitive graph.

    Each role determines how the node participates in activation propagation:
    - Fact: Verifiable information
    - Observation: Noted information without verification
    - Goal: Target state or objective to achieve
    - Constraint: Limitation or restriction on actions
    - Decision: Chosen course of action
    - ConditionalDependency: Relationship between nodes (if X then Y)
    """

    FACT = "fact"
    OBSERVATION = "observation"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    DECISION = "decision"
    CONDITIONAL_DEPENDENCY = "conditional_dependency"


@dataclass
class Node:
    """Represents a commitment chunk in the cognitive graph.

    Attributes:
        id: Unique identifier for the node
        text: The textual content of the commitment
        role: The meta-role assigned to this commitment
        confidence: Confidence score (0.0 to 1.0) for this commitment
        activation: Current activation level (0.0 to 1.0)
        neighbors: Dictionary mapping neighbor_id -> edge weight
        metadata: Optional metadata (timestamps, evidence, etc.)
        embedding: Optional pre-computed embedding vector
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    role: Role = Role.FACT
    confidence: float = 0.7
    activation: float = 0.0
    neighbors: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def add_neighbor(self, neighbor_id: str, weight: float = 1.0) -> None:
        """Add a neighbor with the given weight.

        Args:
            neighbor_id: ID of the neighboring node
            weight: Edge weight (positive for excitatory, negative for inhibitory)
        """
        self.neighbors[neighbor_id] = weight

    def remove_neighbor(self, neighbor_id: str) -> None:
        """Remove a neighbor connection.

        Args:
            neighbor_id: ID of the neighbor to remove
        """
        self.neighbors.pop(neighbor_id, None)

    def update_activation(self, delta: float) -> None:
        """Update activation level with clamping to [0, 1].

        Args:
            delta: Change in activation (can be positive or negative)
        """
        self.activation = max(0.0, min(1.0, self.activation + delta))

    def reset_activation(self) -> None:
        """Reset activation to zero."""
        self.activation = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert node to dictionary representation.

        Returns:
            Dictionary containing all node data
        """
        return {
            "id": self.id,
            "text": self.text,
            "role": self.role.value,
            "confidence": self.confidence,
            "activation": self.activation,
            "neighbors": self.neighbors,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Create a Node from dictionary representation.

        Args:
            data: Dictionary containing node data

        Returns:
            A new Node instance
        """
        return cls(
            id=data["id"],
            text=data["text"],
            role=Role(data["role"]),
            confidence=data.get("confidence", 0.7),
            activation=data.get("activation", 0.0),
            neighbors=data.get("neighbors", {}),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        return f"Node(id={self.id[:8]}..., role={self.role.value}, activation={self.activation:.2f})"
