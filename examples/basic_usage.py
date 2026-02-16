"""Basic usage example for CogMemory.

Demonstrates paragraph ingestion, querying, and activation propagation.

To use with Groq (FREE Llama 3.1 8B):
1. Get API key: https://console.groq.com/
2. Set GROQ_API_KEY environment variable
3. Run this script

To use with dummy extractor (no API required):
- Set use_dummy_extractor=True below
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports without pip install
sys.path.insert(0, str(Path(__file__).parent.parent))

from cog_memory import CognitiveMemory, Role


def main():
    """Run the basic usage example."""
    # Initialize the cognitive memory system

    # Option 1: Use Groq (FREE, recommended) with Llama 3.1 8B
    # Get API key from: https://console.groq.com/
    if os.getenv("GROQ_API_KEY"):
        memory = CognitiveMemory(
            db_path="./data/lancedb",
            provider="groq",
            model="llama-3.1-8b-instant",
        )
        print("Using Groq with Llama 3.1 8B (FREE)\n")
    # Option 2: Use dummy extractor for testing without API key
    else:
        memory = CognitiveMemory(
            db_path="./data/lancedb",
            use_dummy_extractor=True,
        )
        print("Using dummy extractor (no API key required)\n")

    # Sample product launch scenario
    text = """
    We need to launch the new product by Q3 2024. The target is to achieve 1000 active users by the end of the year.
    The budget is limited to $50,000 for marketing. We decided to launch on Monday instead of Friday.
    If the beta testing goes well, we will proceed with full launch. Marketing should start 2 weeks before launch.
    The team observed that users prefer dark mode interface. We cannot exceed the allocated budget under any circumstances.
    """

    print("=" * 60)
    print("INGESTING TEXT")
    print("=" * 60)

    # Ingest the paragraph
    nodes = memory.ingest_paragraph(text)
    print(f"Extracted {len(nodes)} commitments:\n")

    for node in nodes:
        print(f"  [{node.role.value.upper()}] {node.text}")
        print(f"    Confidence: {node.confidence:.2f}\n")

    print("=" * 60)
    print("SYSTEM STATISTICS")
    print("=" * 60)

    stats = memory.get_stats()
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"By role:")
    for role, count in stats['by_role'].items():
        print(f"  {role}: {count}")
    print(f"Total edges: {stats['total_edges']}")
    print(f"Avg activation: {stats['avg_activation']:.3f}\n")

    print("=" * 60)
    print("QUERYING: 'What are the constraints?'")
    print("=" * 60)

    # Query for constraints
    results = memory.query(
        query_text="What are the constraints and limitations?",
        top_k=5,
    )

    print(f"\nTop {len(results)} results:\n")
    for i, node in enumerate(results, 1):
        print(f"{i}. [{node.role.value.upper()}] {node.text}")
        print(f"   Activation: {node.activation:.3f}, Confidence: {node.confidence:.2f}\n")

    print("=" * 60)
    print("GOAL-DRIVEN QUERY")
    print("=" * 60)

    # Get all goals
    goals = memory.get_goals()
    print(f"Found {len(goals)} goals:\n")

    for goal in goals:
        print(f"  - {goal.text}")

    if goals:
        # Activate first goal and propagate
        goal_ids = [goals[0].id]
        print(f"\nActivating goal: '{goals[0].text}'\n")

        results = memory.query(
            query_text="What decisions and actions relate to this goal?",
            top_k=5,
            activate_goals=goal_ids,
        )

        print(f"Top {len(results)} activated nodes:\n")
        for i, node in enumerate(results, 1):
            print(f"{i}. [{node.role.value.upper()}] {node.text}")
            print(f"   Activation: {node.activation:.3f}\n")

    print("=" * 60)
    print("APPLYING DECAY")
    print("=" * 60)

    # Apply decay to simulate time passing
    decay_stats = memory.apply_decay()
    print(f"Nodes decayed: {decay_stats['nodes_decayed']}")
    print(f"Edges pruned: {decay_stats['edges_pruned']}")
    print(f"Inactive nodes: {decay_stats['inactive_nodes']}\n")

    # Show stats after decay
    stats_after = memory.get_stats()
    print(f"Avg activation after decay: {stats_after['avg_activation']:.3f}")


if __name__ == "__main__":
    main()
