"""Deduplication engine for semantic merging.

Detects and merges semantically similar nodes to maintain graph sparsity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cog_memory.node import Node

if TYPE_CHECKING:
    from cog_memory.lance_store import LanceStore


class DeduplicationEngine:
    """Handles deduplication and merging of similar nodes.

    Compares new nodes against existing nodes using similarity search
    and merges or links them when appropriate.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        top_k: int = 5,
    ) -> None:
        """Initialize the deduplication engine.

        Args:
            similarity_threshold: Minimum similarity to trigger merge
            top_k: Number of similar nodes to check
        """
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    def find_similar(
        self,
        node: Node,
        embedding: list[float],
        store: LanceStore,
    ) -> list[dict]:
        """Find similar nodes in the store.

        Args:
            node: Node to compare
            embedding: Embedding of the node
            store: LanceDB store to search

        Returns:
            List of similar node records
        """
        results = store.query_similar(
            embedding=embedding,
            k=self.top_k,
            filter_role=node.role.value if node.role else None,
        )

        # Filter by similarity threshold
        return [
            r for r in results if r.get("similarity", 0) >= self.similarity_threshold
        ]

    def should_merge(
        self,
        node: Node,
        similar_node: dict,
    ) -> bool:
        """Determine if a node should be merged with a similar node.

        Args:
            node: New node
            similar_node: Existing node record from LanceDB

        Returns:
            True if nodes should be merged
        """
        # High similarity + same role = merge
        if similar_node.get("similarity", 0) > 0.95:
            return True

        # For facts and observations, be more conservative
        if node.role in ["fact", "observation"]:
            return similar_node.get("similarity", 0) > 0.95

        return similar_node.get("similarity", 0) >= self.similarity_threshold

    def merge_nodes(
        self,
        existing: dict,
        new: Node,
    ) -> dict:
        """Merge a new node into an existing node.

        Args:
            existing: Existing node record
            new: New node to merge

        Returns:
            Updated node record
        """
        # Update confidence (take max)
        existing["confidence"] = max(existing["confidence"], new.confidence)

        # Merge metadata
        existing["metadata"].update(new.metadata)

        # Add activation
        existing["activation"] = max(existing["activation"], new.activation)

        return existing

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        weight: float = 0.5,
    ) -> None:
        """Create a link between two nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            weight: Edge weight
        """
        # This would update the graph structure
        pass

    def process_node(
        self,
        node: Node,
        embedding: list[float],
        store: LanceStore,
    ) -> tuple[bool, str | None]:
        """Process a new node for deduplication.

        Args:
            node: New node to process
            embedding: Embedding of the node
            store: LanceDB store

        Returns:
            Tuple of (should_skip, merge_with_id)
            - should_skip: True if node should not be added (merged or duplicate)
            - merge_with_id: ID of node to merge with, if any
        """
        similar_nodes = self.find_similar(node, embedding, store)

        if not similar_nodes:
            return False, None

        # Check if we should merge with any similar node
        for similar in similar_nodes:
            if self.should_merge(node, similar):
                return True, similar["id"]

        return False, None

    def __repr__(self) -> str:
        return f"DeduplicationEngine(threshold={self.similarity_threshold}, k={self.top_k})"
