#!/usr/bin/env python3
"""Test script for debugging query propagation.

Usage:
    python test_query.py "is banana a good food?"
    python test_query.py "banana"
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cog_memory.query_interface import CognitiveMemory


def print_tree(tree, node_id, indent=0):
    """Print tree structure."""
    node_data = tree[node_id]
    prefix = "  " * indent
    layer = node_data["layer"]

    if layer == 1:
        print(f"{prefix}📍 Layer 1: DIRECT MATCH")
        print(f"{prefix}  Text: {node_data.get('text', 'N/A')[:60]}...")
        print(f"{prefix}  Similarity: {node_data.get('parent_similarity', 0):.3f}")
    else:
        parent = node_data.get("parent", "N/A")
        activation = node_data.get("activation", 0)
        delta = node_data.get("activation_delta", 0)
        parent_sim = node_data.get("parent_similarity", 0)
        edge = node_data.get("edge_weight", 0)

        print(f"{prefix}📍 Layer {layer}:")
        print(f"{prefix}  Text: {node_data.get('text', 'N/A')[:60]}...")
        print(f"{prefix}  Activation: {activation:.3f}")
        print(f"{prefix}  Formula: {parent_sim:.3f} × {edge:.3f} = {delta:.3f}")
        print(f"{prefix}  Parent: {parent[:8]}...")

    if node_data["children"]:
        for child_id in node_data["children"]:
            print_tree(tree, child_id, indent + 1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_query.py <query>")
        print("Example: python test_query.py \"is banana a good food?\"")
        sys.exit(1)

    query = sys.argv[1]

    print(f"🔍 Query: {query}")
    print("=" * 80)

    # Initialize memory
    memory = CognitiveMemory()

    # Print settings
    print(f"\n⚙️  Settings:")
    print(f"  Min Delta (Gate 1): {memory.graph.min_delta}")
    print(f"  Activation Threshold (Gate 2): {memory.graph.activation_threshold}")

    # Run debug query
    print(f"\n🔄 Running query...")
    result = memory.query_debug(
        query_text=query,
        top_k=10,
        propagation_depth=3,
        min_similarity_threshold=0.55,
    )

    # Print summary
    print(f"\n📊 Results:")
    print(f"  Direct matches (Layer 1): {result['direct_matches_count']}")
    print(f"  Total activated: {result['total_activated']}")
    print(f"  Layers reached: {result['layers']}")

    # Print Layer 1 matches
    print(f"\n📍 Layer 1 (Direct Matches):")
    for match in result["direct_matches"]:
        print(f"  ⚡ {match['similarity_to_query']:.3f} | {match['role']} | {match['text'][:50]}...")

    # Print full tree
    print(f"\n🌳 Propagation Tree:")
    root_ids = [nid for nid, data in result["tree"].items() if data["layer"] == 1]
    for root_id in root_ids:
        print_tree(result["tree"], root_id)

    # Print all activated nodes by layer
    print(f"\n📋 All Activated Nodes by Layer:")
    for layer in sorted(result["layers"].keys()):
        count = result["layers"][layer]
        print(f"  Layer {layer}: {count} nodes")

    # Print detailed tree JSON for inspection
    print(f"\n🔧 Full Tree JSON:")
    print(json.dumps(
        {k: {**v, "children": len(v.get("children", []))} for k, v in result["tree"].items()},
        indent=2,
        default=str
    ))


if __name__ == "__main__":
    main()
