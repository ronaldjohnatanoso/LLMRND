"""Tests for Node class."""

import pytest

from cog_memory.node import Node, Role


def test_node_creation():
    """Test basic node creation."""
    node = Node(
        text="The sky is blue",
        role=Role.FACT,
        confidence=0.9,
    )
    assert node.text == "The sky is blue"
    assert node.role == Role.FACT
    assert node.confidence == 0.9
    assert node.activation == 0.0


def test_node_neighbors():
    """Test neighbor management."""
    node = Node(text="Test", role=Role.FACT)

    node.add_neighbor("neighbor_1", weight=1.0)
    node.add_neighbor("neighbor_2", weight=0.5)

    assert "neighbor_1" in node.neighbors
    assert node.neighbors["neighbor_1"] == 1.0
    assert node.neighbors["neighbor_2"] == 0.5

    node.remove_neighbor("neighbor_1")
    assert "neighbor_1" not in node.neighbors


def test_node_activation():
    """Test activation updates."""
    node = Node(text="Test", role=Role.FACT)

    node.update_activation(0.5)
    assert node.activation == 0.5

    node.update_activation(0.6)
    assert node.activation == 1.0  # Clamped

    node.update_activation(-1.5)
    assert node.activation == 0.0  # Clamped


def test_node_to_dict():
    """Test node serialization."""
    node = Node(
        id="test_id",
        text="Test",
        role=Role.GOAL,
        confidence=0.8,
        metadata={"key": "value"},
    )

    data = node.to_dict()
    assert data["id"] == "test_id"
    assert data["text"] == "Test"
    assert data["role"] == "goal"
    assert data["metadata"] == {"key": "value"}


def test_node_from_dict():
    """Test node deserialization."""
    data = {
        "id": "test_id",
        "text": "Test",
        "role": "decision",
        "confidence": 0.7,
        "activation": 0.5,
        "neighbors": {"n1": 0.8},
        "metadata": {},
    }

    node = Node.from_dict(data)
    assert node.id == "test_id"
    assert node.role == Role.DECISION
    assert node.activation == 0.5
