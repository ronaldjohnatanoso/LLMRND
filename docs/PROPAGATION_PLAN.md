# Dynamic Graph Memory for LLMs - Technical Paper

## 1. Purpose / Goal

Build a memory system for LLMs that:

- **Stores chunks of knowledge** as semantic nodes
- **Retrieves associated chunks** via spreading activation
- **Handles new info without redundancy** through merge/update logic
- **Supports query-time spreading activation** for relevance

## 2. Architecture Overview

### Components

#### Chunk Node
Represents a fact, paragraph, or piece of context

**Attributes:**
- `embedding`: Vector representation
- `role`: Type (FACT, GOAL, DECISION, CONSTRAINT, OBSERVATION, CONDITIONAL_DEPENDENCY)
- `activation`: Current relevance (query-time)
- `neighbors`: Adjacency list of connected nodes

#### Graph Structure
Edges connect related nodes

**Properties:**
- Undirected (simpler, allows associative recall)
- Edge weight = cosine similarity × role boost

**Propagation:**
- Query activates node(s)
- Activation propagates to neighbors with decay
- Only nodes above threshold propagate (active frontier)
- Hop limit / stability check to stop propagation

#### Merge / Update
New chunk embedding compared with existing nodes

**Decision logic:**
- Use separation margin / cosine similarity
- Decide: **merge**, **update**, or **create new**

**Update formula:**
```
V_old = normalize((1 - α) × V_old + α × V_new)
```

Where:
- `V_old` = existing embedding
- `V_new` = new embedding
- `α` = blending factor (how much new info changes old embedding)

#### Retrieval
- Top-K nodes by activation + similarity to query
- Optional margin-based inclusion beyond strict top-K if margin is small (uncertainty)

## 3. Propagation Mechanics

### BFS Traversal
Query-time activation propagation from starting nodes

### Activation Accumulation
```
A_child += A_parent × edge_weight × decay
```

Where:
- `A_child` = child node activation
- `A_parent` = parent node activation
- `edge_weight` = cosine similarity × role boost
- `decay` = per-hop decay factor

### Key Concepts

**Per-hop decay:**
- Decay applied at each edge traversal
- Prevents runaway activation in loops
- Naturally prioritizes near nodes

**Active frontier:**
- Only nodes above threshold can propagate
- Reduces computation
- Ensures quality over quantity

**Stability check:**
```
if max_delta(frontier) < epsilon:
    break  # System stabilized
```

Stop propagation when activation changes are minimal

### Loop Handling
Loops (A ↔ B ↔ A) handled naturally via:
- Decay per hop
- Hop limit
- Stability check

No special "don't propagate back" logic needed

## 4. Node Activation & Relevance

**Activation value = proxy for relevance**

Properties:
- Close activation values → similar relevance/intent
- Reinforcement via multiple parent contributions
- Convergence ensures top-k is meaningful

**Multiple parents:**
When a node receives activation from multiple parents:
```
node.activation = MAX(all_incoming_signals)
```

This preserves the strongest signal rather than summing (prevents overcounting)

## 5. Handling New Chunks

### Ingestion Pipeline

1. **Compute embedding** for new chunk
2. **Compare to existing nodes** via cosine similarity
3. **Decision:**
   - `similarity > threshold` → merge/update
   - `else` → create new node
4. **Update graph edges** based on semantic similarity
5. **Optional:** Propagate activation to update importance dynamically

### Merge / Update Logic

**Current implementation:**
```python
if similarity > 0.95:
    merge  # Nearly identical, merge into existing
elif similarity >= 0.75:
    update  # Related, update confidence
else:
    create_new  # Distinct, create new node
```

**Future: Blending formula**
```python
# Update existing node's embedding with new info
V_old = normalize((1 - alpha) * V_old + alpha * V_new)

# Alpha determines how much new info influences old
alpha = 0.3  # 30% new, 70% old (conservative)
```

## 6. Decay & Stability

### Purpose
- **Decay** prevents runaway loops
- **Hop limit** ensures finite computation
- **Stability check** stops when converged

### Implementation

**Per-hop decay:**
```python
DECAY_PER_HOP = 0.7  # Retain 70% per hop
decay_factor = DECAY_PER_HOP ** current_depth
```

**Stability:**
```python
STABILITY_EPSILON = 0.001  # Max delta to consider stable

# After each hop, check:
max_delta = max(abs(new_activation - old_activation) for all_nodes)
if max_delta < STABILITY_EPSILON:
    break  # System stabilized
```

**Active nodes only:**
Only track nodes with `activation > threshold` for stability checks

## 7. Role / Meta Boost

Each node carries a role that influences propagation

**Role boost matrix:**
```python
ROLE_BOOSTS = {
    (Role.GOAL, Role.DECISION): 1.5,           # Goal → Decision
    (Role.CONSTRAINT, Role.DECISION): -0.8,     # Constraint → Decision
    (Role.OBSERVATION, Role.DECISION): 1.2,     # Observation → Decision
    (Role.FACT, Role.FACT): 1.3,                # Fact reinforcement
    ...
}
```

**Effect:**
Prioritizes certain paths (e.g., goals activate decisions strongly)

## 8. Design Intuition

**Undirected vs Directed:**
- Undirected = simpler, natural associative recall
- Directed = encode causality (future enhancement)

**Query-time only:**
- No retraining required
- Fast, dynamic relevance

**Spreading activation:**
- Mimics biological associative memory
- Allows fuzzy, context-dependent retrieval

**Merge / update:**
- Prevents memory bloat
- Adapts to new information

**Activation + similarity:**
- Enables fuzzy associative recall
- Balances novelty (activation) with relevance (similarity)

## 9. Example Pseudocode (Query-Time)

```python
def query_memory(graph, query_embedding, top_k=10, max_hops=5):
    # Initialize activation from query
    active_nodes = initialize_activation(query_embedding)
    frontier = active_nodes.copy()

    for hop in range(max_hops):
        next_frontier = []

        # Propagate from current frontier
        for node in frontier:
            for neighbor in node.neighbors:
                # Calculate contribution
                edge_weight = cosine(node.embedding, neighbor.embedding)
                role_boost = ROLE_BOOSTS.get((node.role, neighbor.role), 1.0)
                decay = DECAY_PER_HOP ** hop

                contribution = (
                    node.activation *
                    edge_weight *
                    role_boost *
                    decay
                )

                # Accumulate activation
                neighbor.activation = max(
                    neighbor.activation,
                    contribution  # MAX preserves strongest signal
                )

                # Add to next frontier if above threshold
                if neighbor.activation > ACTIVATION_THRESHOLD:
                    next_frontier.append(neighbor)

        # Check stability
        max_delta = compute_max_delta(active_nodes)
        if max_delta < STABILITY_EPSILON:
            break  # Converged

        frontier = next_frontier

    # Return top-k by activation
    return top_k_nodes_by_activation(graph, top_k)
```

## 10. Parameter Reference

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `DECAY_PER_HOP` | 0.7 | 0.5-0.95 | Retention per hop |
| `STABILITY_EPSILON` | 0.001 | 0.0001-0.01 | Max delta for stability |
| `ACTIVATION_THRESHOLD` | 0.5 | 0.1-0.9 | Min activation to propagate |
| `MIN_DELTA` | 0.3 | 0.1-0.5 | Min signal to activate child |
| `MAX_HOPS` | 5 | 3-10 | Hard hop limit |
| `MERGE_THRESHOLD` | 0.95 | 0.9-0.99 | Similarity to merge |
| `UPDATE_THRESHOLD` | 0.75 | 0.7-0.9 | Similarity to update |
| `ALPHA` (future) | 0.3 | 0.1-0.5 | Embedding blend factor |

## 11. Future Enhancements

1. **Blending factor (α)**: Smooth embedding updates for merged nodes
2. **Margin-based retrieval**: Include nodes just below top-K if margin is small
3. **Directed edges**: Encode causal relationships
4. **Temporal decay**: Time-based activation reduction
5. **Importance weighting**: Learn edge weights from usage

## 12. Key Insights

1. **Simplicity first**: Undirected graph + per-hop decay = most of the value
2. **Query-time activation**: No training, fully dynamic
3. **Merge over duplicate**: Prevents memory bloat
4. **Stability matters**: Stop early when converged
5. **Role-based boosting**: Simple way to encode domain knowledge
6. **MAX over SUM**: Preserves strongest signal, prevents overcounting
