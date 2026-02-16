"""Activation decay and forgetting mechanism.

Reduces activation over time and prunes weak connections.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cog_memory.node import Node, Role

if TYPE_CHECKING:
    from collections.abc import Mapping


# Decay rates per role (higher = faster decay)
ROLE_DECAY_RATES: Mapping[Role, float] = {
    Role.FACT: 0.01,  # Facts decay slowly
    Role.OBSERVATION: 0.05,  # Observations decay moderately
    Role.GOAL: 0.02,  # Goals persist
    Role.CONSTRAINT: 0.01,  # Constraints persist
    Role.DECISION: 0.03,  # Decisions decay moderately
    Role.CONDITIONAL_DEPENDENCY: 0.08,  # Dependencies decay faster
}


class DecayModule:
    """Manages activation decay and edge pruning.

    Implements forgetting by reducing activation levels and
    removing weak connections over time.
    """

    def __init__(
        self,
        decay_rate: float = 0.1,
        edge_threshold: float = 0.1,
        activation_threshold: float = 0.01,
    ) -> None:
        """Initialize the decay module.

        Args:
            decay_rate: Base decay rate (multiplied by role-specific rate)
            edge_threshold: Minimum edge weight to keep
            activation_threshold: Threshold for considering a node inactive
        """
        self.decay_rate = decay_rate
        self.edge_threshold = edge_threshold
        self.activation_threshold = activation_threshold

    def decay_node(self, node: Node) -> None:
        """Apply decay to a single node.

        Args:
            node: Node to decay
        """
        role_rate = ROLE_DECAY_RATES.get(node.role, 0.05)
        decay_amount = self.decay_rate * role_rate

        node.activation = max(
            0.0,
            node.activation - decay_amount
        )

    def decay_graph(self, nodes: list[Node]) -> None:
        """Apply decay to all nodes in the graph.

        Args:
            nodes: List of nodes to decay
        """
        for node in nodes:
            self.decay_node(node)

    def prune_weak_edges(self, nodes: list[Node]) -> dict[str, list[str]]:
        """Remove edges below the threshold.

        Args:
            nodes: List of nodes to prune

        Returns:
            Dictionary mapping node_id to list of removed neighbor IDs
        """
        removed = {}

        for node in nodes:
            removed_neighbors = []

            for neighbor_id, weight in list(node.neighbors.items()):
                if abs(weight) < self.edge_threshold:
                    node.remove_neighbor(neighbor_id)
                    removed_neighbors.append(neighbor_id)

            if removed_neighbors:
                removed[node.id] = removed_neighbors

        return removed

    def get_inactive_nodes(self, nodes: list[Node]) -> list[Node]:
        """Get nodes below the activation threshold.

        Args:
            nodes: List of nodes to check

        Returns:
            List of inactive nodes
        """
        return [
            node for node in nodes
            if node.activation < self.activation_threshold
        ]

    def boost_node(self, node: Node, amount: float = 0.2) -> None:
        """Boost activation of a node (opposite of decay).

        Useful for reinforcing important information.

        Args:
            node: Node to boost
            amount: Amount to boost by
        """
        node.activation = min(1.0, node.activation + amount)

    def __repr__(self) -> str:
        return f"DecayModule(rate={self.decay_rate}, edge_thresh={self.edge_threshold})"
