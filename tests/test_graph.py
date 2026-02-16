"""Tests for CognitiveGraph class."""

import pytest

from cog_memory.cognitive_graph import CognitiveGraph
from cog_memory.node import Node, Role


def test_graph_creation():
    """Test graph initialization."""
    graph = CognitiveGraph()
    assert len(graph) == 0
    assert graph.propagation_depth == 2


def test_add_node():
    """Test adding nodes to graph."""
    graph = CognitiveGraph()
    node = Node(text="Test", role=Role.FACT)

    graph.add_node(node)
    assert len(graph) == 1
    assert graph.get_node(node.id) == node


def test_activate_node():
    """Test node activation."""
    graph = CognitiveGraph()
    node = Node(text="Test", role=Role.FACT)

    graph.add_node(node)
    graph.activate_node(node.id, 0.8)

    assert node.activation == 0.8


def test_edge_creation():
    """Test edge creation between nodes."""
    graph = CognitiveGraph()
    node1 = Node(text="Node 1", role=Role.FACT)
    node2 = Node(text="Node 2", role=Role.DECISION)

    graph.add_node(node1)
    graph.add_node(node2)
    graph.add_edge(node1.id, node2.id, weight=1.5)

    assert node2.id in node1.neighbors
    assert node1.neighbors[node2.id] == 1.5


def test_activation_propagation():
    """Test activation propagation through graph."""
    graph = CognitiveGraph(propagation_depth=2)

    # Create a chain: goal -> decision -> fact
    goal = Node(text="Goal", role=Role.GOAL)
    decision = Node(text="Decision", role=Role.DECISION)
    fact = Node(text="Fact", role=Role.FACT)

    graph.add_node(goal)
    graph.add_node(decision)
    graph.add_node(fact)

    # Create edges
    graph.add_edge(goal.id, decision.id, weight=1.0)
    graph.add_edge(decision.id, fact.id, weight=1.0)

    # Activate goal
    graph.activate_node(goal.id, 1.0)
    graph.propagate_activation(goal.id)

    # Decision should be activated (with role boost)
    assert decision.activation > 0
    # Fact should also be activated
    assert fact.activation > 0


def test_get_top_activated():
    """Test retrieving top activated nodes."""
    graph = CognitiveGraph()

    for i in range(5):
        node = Node(text=f"Node {i}", role=Role.FACT, activation=i * 0.2)
        graph.add_node(node)

    top = graph.get_top_activated(3)
    assert len(top) == 3
    assert top[0].activation == 0.8
    assert top[2].activation == 0.4


def test_get_nodes_by_role():
    """Test filtering nodes by role."""
    graph = CognitiveGraph()

    graph.add_node(Node(text="F1", role=Role.FACT))
    graph.add_node(Node(text="G1", role=Role.GOAL))
    graph.add_node(Node(text="F2", role=Role.FACT))

    facts = graph.get_nodes_by_role(Role.FACT)
    goals = graph.get_nodes_by_role(Role.GOAL)

    assert len(facts) == 2
    assert len(goals) == 1


def test_reset_activations():
    """Test resetting all activations."""
    graph = CognitiveGraph()

    node1 = Node(text="N1", role=Role.FACT, activation=0.8)
    node2 = Node(text="N2", role=Role.GOAL, activation=0.5)

    graph.add_node(node1)
    graph.add_node(node2)

    graph.reset_all_activations()

    assert node1.activation == 0.0
    assert node2.activation == 0.0
