"""Neuroplasticity manager for Hebbian learning and connection weight updates.

Implements "cells that fire together, wire together" learning where connections
strengthen based on usage patterns, mimicking synaptic plasticity.
"""

import json
from pathlib import Path
from typing import Any

from cog_memory.cognitive_graph import CognitiveGraph
from cog_memory.plasticity_config import PlasticityConfig


class NeuroplasticityManager:
    """Manages connection weight updates based on usage patterns.

    Tracks propagation history and applies Hebbian learning to strengthen
    frequently-used connections and prune weak ones.
    """

    def __init__(self, graph: CognitiveGraph, config: PlasticityConfig):
        """Initialize the neuroplasticity manager.

        Args:
            graph: The cognitive graph to manage
            config: Plasticity configuration
        """
        self.graph = graph
        self.config = config
        self.query_count = 0
        self.last_consolidation_query = 0

    def learn_from_query(self, propagation_history: list[dict]) -> dict:
        """Apply Hebbian learning based on propagation patterns.

        Args:
            propagation_history: List of propagation events from a query
                Each event should have: parent_id, child_id, delta, hop

        Returns:
            Statistics about what was learned
        """
        self.query_count += 1
        stats = {
            "connections_strengthened": 0,
            "connections_weakened": 0,
            "total_weight_change": 0.0,
            "consolidation_ran": False,
        }

        for event in propagation_history:
            parent_id = event.get("parent_id")
            child_id = event.get("child_id")
            activation_delta = event.get("delta", 0.0)

            if not parent_id or not child_id:
                continue

            parent = self.graph.get_node(parent_id)
            if not parent or child_id not in parent.neighbors:
                continue

            old_weight = parent.neighbors[child_id]

            # Hebbian learning: strengthen based on signal strength
            if self.config.learning_mode == "hebbian":
                weight_change = activation_delta * self.config.learning_rate
                new_weight = old_weight + weight_change

            # Apply constraints
            new_weight = max(
                self.config.min_weight, min(self.config.max_weight, new_weight)
            )

            # Update weight
            parent.neighbors[child_id] = new_weight

            if new_weight > old_weight:
                stats["connections_strengthened"] += 1
            else:
                stats["connections_weakened"] += 1

            stats["total_weight_change"] += abs(new_weight - old_weight)

        # Periodic consolidation
        if self.config.consolidation_enabled:
            if (
                self.query_count - self.last_consolidation_query
            ) >= self.config.consolidation_interval:
                consolidation_stats = self._consolidate_memory()
                stats["consolidation"] = consolidation_stats
                stats["consolidation_ran"] = True
                self.last_consolidation_query = self.query_count

        return stats

    def _consolidate_memory(self) -> dict:
        """Strengthen frequently-used connections, prune weak ones.

        Mimics sleep consolidation where important pathways strengthen
        and irrelevant ones fade away.

        Returns:
            Statistics about consolidation
        """
        stats = {"strengthened": 0, "pruned": 0}

        for node in self.graph.get_all_nodes():
            for neighbor_id, weight in list(node.neighbors.items()):
                usage_count = node.connection_usage.get(neighbor_id, 0)

                # Strengthen frequently used
                if usage_count >= self.config.consolidation_strength_threshold:
                    old_weight = weight
                    new_weight = min(self.config.max_weight, weight * 1.1)
                    node.neighbors[neighbor_id] = new_weight
                    stats["strengthened"] += 1

                # Prune rarely used and weak
                elif (
                    usage_count < self.config.consolidation_prune_threshold
                    and weight < self.config.decay_threshold
                ):
                    del node.neighbors[neighbor_id]
                    if neighbor_id in node.connection_usage:
                        del node.connection_usage[neighbor_id]
                    if neighbor_id in node.connection_last_used:
                        del node.connection_last_used[neighbor_id]
                    stats["pruned"] += 1

        # Reset usage counters after consolidation
        self._reset_usage_counters()

        return stats

    def _reset_usage_counters(self) -> None:
        """Reset connection usage counters after consolidation."""
        for node in self.graph.get_all_nodes():
            node.connection_usage.clear()
            node.connection_last_used.clear()

    def apply_feedback(
        self, propagation_history: list[dict], feedback: str
    ) -> dict:
        """Apply user feedback to adjust connection weights.

        Args:
            propagation_history: List of propagation events from the query
            feedback: Either "positive" or "negative"

        Returns:
            Statistics about connections adjusted
        """
        if not self.config.reinforcement_enabled:
            return {"error": "Reinforcement learning not enabled"}

        stats = {"connections_adjusted": 0}

        if feedback == "positive":
            boost = self.config.positive_feedback_boost
            for event in propagation_history:
                parent = self.graph.get_node(event.get("parent_id", ""))
                if parent and event.get("child_id") in parent.neighbors:
                    child_id = event["child_id"]
                    old_weight = parent.neighbors[child_id]
                    new_weight = min(self.config.max_weight, old_weight + boost)
                    parent.neighbors[child_id] = new_weight
                    stats["connections_adjusted"] += 1

        elif feedback == "negative":
            penalty = self.config.negative_feedback_penalty
            for event in propagation_history:
                parent = self.graph.get_node(event.get("parent_id", ""))
                if parent and event.get("child_id") in parent.neighbors:
                    child_id = event["child_id"]
                    old_weight = parent.neighbors[child_id]
                    new_weight = max(self.config.min_weight, old_weight - penalty)
                    parent.neighbors[child_id] = new_weight
                    stats["connections_adjusted"] += 1

        return stats

    def save_weights(self, filepath: str) -> None:
        """Save learned connection weights to disk.

        Args:
            filepath: Path to save weights JSON file
        """
        data: dict[str, Any] = {
            "metadata": {
                "query_count": self.query_count,
                "last_consolidation_query": self.last_consolidation_query,
            },
            "nodes": {},
        }

        for node in self.graph.get_all_nodes():
            data["nodes"][node.id] = {
                "neighbors": node.neighbors,
                "connection_usage": node.connection_usage,
                "connection_last_used": node.connection_last_used,
            }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_weights(self, filepath: str) -> bool:
        """Load learned connection weights from disk.

        Args:
            filepath: Path to weights JSON file

        Returns:
            True if weights were loaded, False if file doesn't exist
        """
        if not Path(filepath).exists():
            return False

        with open(filepath) as f:
            data = json.load(f)

        # Load metadata
        metadata = data.get("metadata", {})
        self.query_count = metadata.get("query_count", 0)
        self.last_consolidation_query = metadata.get(
            "last_consolidation_query", 0
        )

        # Load node weights
        for node_id, node_data in data.get("nodes", {}).items():
            node = self.graph.get_node(node_id)
            if node:
                node.neighbors = node_data.get("neighbors", {})
                node.connection_usage = node_data.get("connection_usage", {})
                node.connection_last_used = node_data.get(
                    "connection_last_used", {}
                )

        return True
