"""Tests for DeduplicationEngine class."""

import pytest

from cog_memory.deduplication import DeduplicationEngine
from cog_memory.node import Node, Role


def test_deduplication_init():
    """Test deduplication engine initialization."""
    engine = DeduplicationEngine(
        similarity_threshold=0.9,
        top_k=3,
    )

    assert engine.similarity_threshold == 0.9
    assert engine.top_k == 3


def test_should_merge_high_similarity():
    """Test merging decision for high similarity nodes."""
    engine = DeduplicationEngine(similarity_threshold=0.85)

    node1 = Node(text="The sky is blue", role=Role.FACT)
    similar_record = {
        "id": "existing_id",
        "text": "The sky is blue",
        "role": "fact",
        "similarity": 0.96,
        "confidence": 0.9,
        "activation": 0.0,
        "neighbors": {},
        "metadata": {},
    }

    assert engine.should_merge(node1, similar_record) is True


def test_should_merge_low_similarity():
    """Test merging decision for low similarity nodes."""
    engine = DeduplicationEngine(similarity_threshold=0.85)

    node1 = Node(text="The sky is blue", role=Role.FACT)
    similar_record = {
        "id": "existing_id",
        "text": "Grass is green",
        "role": "fact",
        "similarity": 0.75,
        "confidence": 0.9,
        "activation": 0.0,
        "neighbors": {},
        "metadata": {},
    }

    assert engine.should_merge(node1, similar_record) is False


def test_merge_nodes():
    """Test node merging."""
    engine = DeduplicationEngine()

    existing = {
        "id": "existing_id",
        "text": "Test",
        "role": "fact",
        "confidence": 0.7,
        "activation": 0.3,
        "neighbors": {},
        "metadata": {"source": "original"},
    }

    new_node = Node(
        id="new_id",
        text="Test",
        role=Role.FACT,
        confidence=0.9,
        metadata={"source": "new"},
    )

    merged = engine.merge_nodes(existing, new_node)

    assert merged["confidence"] == 0.9  # Max of both
    assert merged["activation"] == 0.3  # Max of both
    assert merged["metadata"]["source"] == "new"  # Updated
