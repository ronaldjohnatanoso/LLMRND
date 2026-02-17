# CogMemory System Flow Analysis

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT TEXT                                │
│                  "The system requires 50TB..."                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. LLM EXTRACTOR (llm_extractor.py)                            │
│                                                                 │
│  Extracts commitments with meta-roles:                          │
│  • FACT - Objective facts                                       │
│  • OBSERVATION - Empirical observations                          │
│  • GOAL - Objectives/targets                                    │
│  • CONSTRAINT - Restrictions/requirements                       │
│  • DECISION - Choices made                                      │
│  • CONDITIONAL_DEPENDENCY - If-then relationships               │
│                                                                 │
│  Output: List[Node] with role assignments                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. EMBEDDING MANAGER (embedding_manager.py)                    │
│                                                                 │
│  Generates vector embeddings:                                   │
│  • Uses Nomic API (768 dim) by default                          │
│  • Batch processing for efficiency                              │
│  • Stores in node.embedding field                               │
│                                                                 │
│  Output: List[vector] - one per node                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. SEMANTIC EDGE CREATION (query_interface.py)                 │
│                                                                 │
│  Creates edges based on cosine similarity:                      │
│  • Between new nodes: similarity ≥ 0.55 → edge                  │
│  • To existing nodes: query top-3 similar → if ≥ 0.55 → edge   │
│  • Edge weight = similarity score                              │
│                                                                 │
│  ✅ WORKING: Edges are created during ingestion                  │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. DEDUPLICATION (deduplication.py)                            │
│                                                                 │
│  Checks for duplicates before storing:                          │
│  • Queries similar nodes from LanceDB                          │
│  • If similarity ≥ 0.90 → merge or skip                        │
│  • Updates confidence on merge                                 │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. PERSISTENT STORAGE (lance_store.py)                         │
│                                                                 │
│  LanceDB vector database:                                       │
│  • Stores: id, text, role, confidence, activation, neighbors    │
│  • Vector column for similarity search                          │
│  • Persistent to disk at ./data/lancedb                         │
│                                                                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. IN-MEMORY GRAPH (cognitive_graph.py)                        │
│                                                                 │
│  Maintains working graph:                                       │
│  • Nodes loaded from LanceDB on startup                        │
│  • Adjacency list via node.neighbors dict                      │
│  • Activation propagation methods ✅ EXISTS                     │
│  • Role-based boost weights defined                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. QUERY PROCESSING (query_interface.py) ❌ BROKEN            │
│                                                                 │
│  Current behavior:                                              │
│  1. Reset all activations to 0 ✅                               │
│  2. Generate query embedding ✅                                 │
│  3. Find similar nodes via vector search ✅                     │
│  4. Activate direct matches ✅                                  │
│  5. Return top activated ❌ NO PROPAGATION!                     │
│                                                                 │
│  MISSING:                                                       │
│  • No propagation to neighbors                                 │
│  • No role-based boosting                                      │
│  • No multi-hop activation spread                              │
│                                                                 │
│  What SHOULD happen:                                            │
│  1. Activate direct matches                                     │
│  2. For each activated node:                                    │
│     → propagate_activation(node_id, depth=2)                   │
│     → Uses role boosts: GOAL→DECISION ×1.5, etc.               │
│  3. Return top activated (includes propagated nodes)           │
└─────────────────────────────────────────────────────────────────┘
```

## Role-Based Activation Boosts (DEFINED BUT NOT USED)

```python
ROLE_BOOSTS = {
    (GOAL, DECISION): 1.5,              # Goals strongly activate decisions
    (CONSTRAINT, DECISION): -0.8,        # Constraints inhibit decisions
    (OBSERVATION, DECISION): 1.2,        # Observations support decisions
    (FACT, FACT): 1.3,                   # Facts reinforce facts
    (GOAL, CONDITIONAL_DEPENDENCY): 1.2, # Goals activate conditionals
    # ... more rules
}
```

## Propagation Algorithm (EXISTS BUT NOT CALLED)

```python
def propagate_activation(self, start_node_id, depth=2):
    """Spread activation from node to neighbors."""
    queue = [(start_node_id, 0)]

    while queue:
        node_id, current_depth = queue.pop(0)

        for neighbor_id, edge_weight in node.neighbors.items():
            # Calculate boost based on roles
            role_boost = ROLE_BOOSTS.get((node.role, neighbor.role), 1.0)

            # Apply activation
            activation_delta = node.activation × edge_weight × role_boost
            neighbor.update_activation(activation_delta)

            # Continue propagating if depth < max
            if current_depth + 1 < depth:
                queue.append((neighbor_id, current_depth + 1))
```

## Issues Found

### ❌ CRITICAL: Propagation Not Used

**Problem:** `query()` never calls `propagate_activation()`

**Impact:**
- Activations only reflect direct vector similarity
- No multi-hop reasoning
- Role-based boosts ignored
- Network structure unused during query

**Fix:** Add propagation in query_interface.py:147

### ✅ WORKING: Edge Creation

- Edges created during ingestion with similarity ≥ 0.55
- Connects new nodes to each other
- Connects new nodes to existing similar nodes
- Edge weights stored correctly

### ✅ WORKING: Everything Else

- LLM extraction with role assignment
- Embedding generation (Nomic 768 dim)
- LanceDB persistence
- Graph structure maintenance
- Deduplication

## Recommended Fix

In `query_interface.py`, line 180, add:

```python
# Activate similar nodes
activated_ids = set()
for record in similar_records[:top_k]:
    node_id = record["id"]
    similarity = record.get("similarity", 0.5)
    self.graph.activate_node(node_id, activation=similarity)
    activated_ids.add(node_id)

# ✅ ADD THIS: Propagate activation through graph
for node_id in activated_ids:
    self.graph.propagate_activation(node_id, depth=2)
```

This would make activations meaningful by spreading energy through the network!
