"""Cognitive graph with activation propagation.

The cognitive graph maintains nodes and their connections, supporting
goal-driven activation with role-based modulation.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

from cog_memory.node import Node, Role

if TYPE_CHECKING:
    from collections.abc import Mapping


# Role-to-role activation boost weights
# Format: (source_role, target_role) -> boost_multiplier
ROLE_BOOSTS: Mapping[tuple[Role, Role], float] = {
    # Goal activates decisions
    (Role.GOAL, Role.DECISION): 1.5,
    # Constraint inhibits conflicting decisions
    (Role.CONSTRAINT, Role.DECISION): -0.8,
    # Observation supports decisions
    (Role.OBSERVATION, Role.DECISION): 1.2,
    # Facts reinforce related facts
    (Role.FACT, Role.FACT): 1.3,
    # Conditional dependencies bridge to decisions/constraints
    (Role.CONDITIONAL_DEPENDENCY, Role.DECISION): 1.1,
    (Role.CONDITIONAL_DEPENDENCY, Role.CONSTRAINT): 1.1,
    # Goals activate conditional dependencies
    (Role.GOAL, Role.CONDITIONAL_DEPENDENCY): 1.2,
    # Default (same-role activation)
    (Role.FACT, Role.FACT): 1.0,
    (Role.OBSERVATION, Role.OBSERVATION): 1.0,
}


class CognitiveGraph:
    """Sparse cognitive graph with activation propagation.

    The graph stores nodes and their connections in memory, supporting
    multi-hop activation propagation with role-based modulation.
    """

    def __init__(
        self,
        propagation_depth: int = 2,
        activation_threshold: float = 0.5,
        default_boost: float = 1.0,
        min_delta: float = 0.3,
    ) -> None:
        """Initialize the cognitive graph.

        Args:
            propagation_depth: Maximum depth for activation propagation
            activation_threshold: Minimum activation to continue propagation (default: 0.5)
            default_boost: Default boost multiplier when no role-specific rule exists
            min_delta: Minimum activation delta to propagate to children (default: 0.3)
        """
        self.nodes: dict[str, Node] = {}
        self.propagation_depth = propagation_depth
        self.activation_threshold = activation_threshold
        self.default_boost = default_boost
        self.min_delta = min_delta

    def add_node(self, node: Node) -> None:
        """Add a node to the graph.

        Args:
            node: Node to add
        """
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID.

        Args:
            node_id: ID of the node to retrieve

        Returns:
            Node if found, None otherwise
        """
        return self.nodes.get(node_id)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph.

        Also removes all edges pointing to this node.

        Args:
            node_id: ID of the node to remove
        """
        if node_id in self.nodes:
            del self.nodes[node_id]
            # Remove edges pointing to this node
            for node in self.nodes.values():
                node.remove_neighbor(node_id)

    def add_edge(self, source_id: str, target_id: str, weight: float = 1.0) -> None:
        """Add a directed edge between nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            weight: Edge weight (positive for excitatory, negative for inhibitory)
        """
        source = self.get_node(source_id)
        if source:
            source.add_neighbor(target_id, weight)

    def activate_node(self, node_id: str, activation: float = 1.0) -> None:
        """Activate a specific node.

        Args:
            node_id: ID of the node to activate
            activation: Activation level to set (0.0 to 1.0)
        """
        node = self.get_node(node_id)
        if node:
            node.activation = max(0.0, min(1.0, activation))

    def propagate_activation(self, start_node_id: str, depth: int | None = None) -> None:
        """Propagate activation from a starting node through the graph.

        Uses BFS traversal with role-based boost modulation.

        Args:
            start_node_id: ID of the node to start propagation from
            depth: Maximum propagation depth (uses instance default if None)
        """
        if depth is None:
            depth = self.propagation_depth

        start_node = self.get_node(start_node_id)
        if not start_node or start_node.activation < self.activation_threshold:
            return

        # Use queue for BFS with (node_id, current_depth)
        queue = deque([(start_node_id, 0)])
        visited = set()

        while queue:
            node_id, current_depth = queue.popleft()

            if current_depth >= depth or node_id in visited:
                continue

            visited.add(node_id)
            node = self.get_node(node_id)

            if not node or node.activation < self.activation_threshold:
                continue

            # Propagate to neighbors
            for neighbor_id, weight in node.neighbors.items():
                neighbor = self.get_node(neighbor_id)
                if not neighbor:
                    continue

                # Calculate role-based boost
                role_boost = ROLE_BOOSTS.get(
                    (node.role, neighbor.role), self.default_boost
                )

                # Use parent's similarity_to_query (not activation) for propagation
                # This prevents stale activation values from affecting calculations
                base_similarity = node.similarity_to_query if node.similarity_to_query > 0 else node.activation
                activation_delta = base_similarity * weight * role_boost

                # Only propagate significant signals (above min_delta threshold)
                if activation_delta < self.min_delta:
                    continue

                neighbor.update_activation(activation_delta)

                # Add to queue if not at max depth AND neighbor will continue propagating
                if current_depth + 1 < depth and neighbor.activation >= self.activation_threshold:
                    queue.append((neighbor_id, current_depth + 1))

    def activate_goals(self, goal_ids: list[str], activation: float = 1.0) -> None:
        """Activate multiple goal nodes and propagate activation.

        Args:
            goal_ids: List of goal node IDs to activate
            activation: Activation level for each goal
        """
        for goal_id in goal_ids:
            self.activate_node(goal_id, activation)

        for goal_id in goal_ids:
            self.propagate_activation(goal_id)

    def reset_all_activations(self) -> None:
        """Reset all node activations and similarity_to_query to zero."""
        for node in self.nodes.values():
            node.reset_activation()
            node.similarity_to_query = 0.0  # Reset similarity tracking too

    def get_top_activated(self, n: int = 10) -> list[Node]:
        """Get the top N most activated nodes.

        Args:
            n: Number of top nodes to return

        Returns:
            List of nodes sorted by activation (descending)
        """
        # Filter out nodes with zero activation
        activated_nodes = [n for n in self.nodes.values() if n.activation > 0]
        sorted_nodes = sorted(
            activated_nodes, key=lambda n: n.activation, reverse=True
        )
        return sorted_nodes[:n]

    def get_nodes_by_role(self, role: Role) -> list[Node]:
        """Get all nodes with a specific role.

        Args:
            role: Role to filter by

        Returns:
            List of nodes with the specified role
        """
        return [node for node in self.nodes.values() if node.role == role]

    def __len__(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"CognitiveGraph(nodes={len(self.nodes)}, depth={self.propagation_depth})"
