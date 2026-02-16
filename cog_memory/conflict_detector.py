"""Conflict and prediction error detection.

Identifies contradictions and conflicting nodes in the cognitive graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cog_memory.node import Node, Role

if TYPE_CHECKING:
    from collections.abc import Sequence


class ConflictDetector:
    """Detects conflicts and contradictions between nodes.

    Uses embedding similarity and negation detection to identify conflicts.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.9,
        enable_llm_check: bool = False,
    ) -> None:
        """Initialize the conflict detector.

        Args:
            similarity_threshold: Minimum similarity to check for conflicts
            enable_llm_check: Use LLM for deeper conflict detection
        """
        self.similarity_threshold = similarity_threshold
        self.enable_llm_check = enable_llm_check

        # Negation words for simple detection
        self.negation_words = {
            "not",
            "no",
            "never",
            "none",
            "cannot",
            "can't",
            "won't",
            "don't",
            "doesn't",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
        }

    def has_negation(self, text: str) -> bool:
        """Check if text contains negation words.

        Args:
            text: Text to check

        Returns:
            True if negation detected
        """
        words = set(text.lower().split())
        return bool(words & self.negation_words)

    def check_conflict(
        self,
        node1: Node,
        node2: Node,
        similarity: float,
    ) -> bool:
        """Check if two nodes conflict with each other.

        Args:
            node1: First node
            node2: Second node
            similarity: Similarity score between nodes

        Returns:
            True if nodes conflict
        """
        # Only high-similarity nodes can conflict
        if similarity < self.similarity_threshold:
            return False

        # Check for direct negation
        has_neg1 = self.has_negation(node1.text)
        has_neg2 = self.has_negation(node2.text)

        # If one has negation and the other doesn't, they might conflict
        if has_neg1 != has_neg2:
            return self._check_semantic_conflict(node1, node2)

        # Check for opposite roles (e.g., constraint vs goal)
        if self._check_role_conflict(node1, node2):
            return True

        return False

    def _check_semantic_conflict(self, node1: Node, node2: Node) -> bool:
        """Check semantic conflict between nodes.

        Args:
            node1: First node
            node2: Second node

        Returns:
            True if semantic conflict detected
        """
        # Simple heuristic: if texts are similar but one has negation
        text1_words = set(node1.text.lower().split())
        text2_words = set(node2.text.lower().split())

        # Remove negation words for comparison
        negation_words = {"not", "no", "never", "none", "cannot", "can't", "won't", "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't"}
        text1_words -= negation_words
        text2_words -= negation_words

        # Check if remaining words overlap significantly
        overlap = len(text1_words & text2_words) / max(len(text1_words | text2_words), 1)

        return overlap > 0.7

    def _check_role_conflict(self, node1: Node, node2: Node) -> bool:
        """Check if roles indicate a conflict.

        Args:
            node1: First node
            node2: Second node

        Returns:
            True if roles suggest conflict
        """
        # Constraints and goals with similar content might conflict
        if {node1.role, node2.role} == {Role.CONSTRAINT, Role.GOAL}:
            return self._check_semantic_conflict(node1, node2)

        return False

    def find_conflicts(
        self,
        node: Node,
        candidates: Sequence[Node],
        similarities: Sequence[float],
    ) -> list[tuple[Node, float]]:
        """Find conflicting nodes among candidates.

        Args:
            node: Node to check conflicts for
            candidates: Potential conflicting nodes
            similarities: Similarity scores for candidates

        Returns:
            List of (conflicting_node, similarity) tuples
        """
        conflicts = []

        for candidate, sim in zip(candidates, similarities):
            if self.check_conflict(node, candidate, sim):
                conflicts.append((candidate, sim))

        return conflicts

    def __repr__(self) -> str:
        return f"ConflictDetector(threshold={self.similarity_threshold}, llm={self.enable_llm_check})"
