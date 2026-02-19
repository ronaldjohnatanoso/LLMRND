# CogMemory Architecture

## System Overview

CogMemory implements a biologically-inspired cognitive memory system that extracts, stores, and retrieves semantic commitments from text. The system combines vector similarity search with graph-based activation propagation for goal-driven reasoning.

## Core Concepts

### Commitments

A commitment is a semantic unit extracted from text representing:
- A fact or observation
- A goal or objective
- A constraint or limitation
- A decision or action
- A conditional dependency

Each commitment is stored as a [Node](../cog_memory/node.py) with:
- Unique identifier
- Text content
- Meta-role classification
- Confidence score
- Activation level
- Neighbor connections (edges)

### Meta-Roles

Meta-roles determine how commitments participate in activation propagation:

```
FACT              → Verifiable information, decays slowly
OBSERVATION       → Noted information, moderate decay
GOAL              → Target state, activates decisions
CONSTRAINT        → Limitation, inhibits conflicting decisions
DECISION          → Chosen action, moderate decay
CONDITIONAL_DEPENDENCY → If-then relationships, fast decay
```

## Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                     CognitiveMemory API                      │
│                   (query_interface.py)                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ LLMExtractor │   │  Cognitive   │   │  LanceStore  │
│              │   │    Graph     │   │              │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                  │
       │                  │                  │
┌──────▼───────┐   ┌──────▼───────┐   ┌──────▼───────┐
│   Embedding  │   │ Deduplication│   │    Decay     │
│   Manager    │   │   Engine     │   │   Module     │
└──────────────┘   └──────────────┘   └──────────────┘
```

## Component Details

### 1. LLM Extractor

**File:** [cog_memory/llm_extractor.py](../cog_memory/llm_extractor.py)

Extracts commitments from text using LLM with structured output:

```
Input: Paragraph text
  ↓
Prompt: Extract commitments with meta-roles
  ↓
LLM: GPT-4o (or dummy extractor)
  ↓
Output: List[Node] with Role assignments
```

**Extraction Prompt Structure:**
- Define meta-roles with descriptions
- Request JSON output format
- Include confidence scoring
- Provide examples

### 2. Embedding Manager

**File:** [cog_memory/embedding_manager.py](../cog_memory/embedding_manager.py)

Generates vector embeddings for semantic similarity:

```
OpenAI text-embedding-3-small  → 1536 dimensions
OpenAI text-embedding-3-large  → 3072 dimensions
sentence-transformers          → 384 dimensions (default)
```

**Batch Processing:**
- Single embedding: `generate_embedding(text)`
- Batch: `generate_embeddings_batch(texts)`
- Efficient for multiple nodes

### 3. LanceDB Store

**File:** [cog_memory/lance_store.py](../cog_memory/lance_store.py)

Persistent vector storage with schema:

```python
schema = {
    "id": string,
    "text": string,
    "role": string,
    "confidence": float,
    "activation": float,
    "neighbors": string (JSON),
    "metadata": string (JSON),
    "vector": float[] (1536 or 3072 dim),
}
```

**Operations:**
- `add_node()` - Insert with embedding
- `query_similar()` - Top-K similarity search
- `update_node()` - Update fields
- `get_all_nodes()` - Load into graph

### 4. Cognitive Graph

**File:** [cog_memory/cognitive_graph.py](../cog_memory/cognitive_graph.py)

In-memory sparse graph with activation propagation:

**Graph Structure:**
```
Node.neighbors: {neighbor_id: weight}
```

**Activation Propagation (BFS with Stability):**
```
propagate_activation(start_nodes, max_hops=5, stability_epsilon=0.001):
    queue = [(start_node, 0)]
    previous_activations = {}
    visited_edges = set()  # Track (parent, child) to prevent cycles

    for hop in range(max_hops):
        current_activations = {n.id: n.activation for n in active_nodes}

        while queue:
            node, current_depth = queue.popleft()

            # Only propagate from sufficiently activated nodes (active frontier)
            if node.activation < min_delta:
                continue

            for neighbor_id, weight in node.neighbors:
                edge = (node.id, neighbor_id)
                if edge in visited_edges:
                    continue  # Prevent cycles
                visited_edges.add(edge)

                # Per-edge decay: activation decays with each hop
                decay_factor = DECAY_PER_HOP ** current_depth
                role_boost = ROLE_BOOSTS[(node.role, neighbor.role)]
                delta = node.activation * weight * role_boost * decay_factor

                # Sum contributions from multiple parents
                neighbor.activation += delta

                queue.append((neighbor, current_depth + 1))

        # Check stability: stop if max delta < epsilon
        max_delta = max(abs(current - previous) for current, previous in
                       zip(current_activations.values(), previous_activations.values()))
        if max_delta < stability_epsilon:
            break  # System stabilized
```

**Key Improvements:**
- **Active Frontier**: Only nodes with activation > `min_delta` can propagate
- **Per-Edge Decay**: Decay applied at each hop (path-based)
- **Stability-Based Stopping**: Stop when activation changes < epsilon
- **Cycle Prevention**: Track visited edges (not just nodes)
- **Multi-Parent Summation**: Nodes receive summed contributions from all parents

**Role Boost Matrix:**
```python
ROLE_BOOSTS = {
    (Role.GOAL, Role.DECISION): 1.5,           # Goal → Decision
    (Role.CONSTRAINT, Role.DECISION): -0.8,     # Constraint → Decision
    (Role.OBSERVATION, Role.DECISION): 1.2,     # Observation → Decision
    (Role.FACT, Role.FACT): 1.3,                # Fact reinforcement
    # ... more rules
}
```

**Propagation Parameters:**
```python
MIN_DELTA = 0.01              # Minimum activation to propagate
DECAY_PER_HOP = 0.7           # Retain 70% per hop
STABILITY_EPSILON = 0.001     # Max delta to consider stable
MAX_HOPS = 5                  # Hard safety limit
```

See [PROPAGATION_PLAN.md](PROPAGATION_PLAN.md) for detailed implementation roadmap.

### 5. Deduplication Engine

**File:** [cog_memory/deduplication.py](../cog_memory/deduplication.py)

Detects and merges semantically similar nodes:

**Pipeline:**
```
1. Query LanceDB for top-K similar nodes
2. Filter by similarity threshold
3. Check should_merge() based on role and similarity
4. If merge: update existing node with max confidence
5. Else: add new node
```

**Merge Logic:**
```python
def should_merge(node, similar_node):
    if similarity > 0.95: return True
    if role in [FACT, OBSERVATION]: return similarity > 0.95
    return similarity >= threshold
```

### 6. Decay Module

**File:** [cog_memory/decay.py](../cog_memory/decay.py)

Implements forgetting and edge pruning:

**Role-Specific Decay Rates:**
```python
ROLE_DECAY_RATES = {
    Role.FACT: 0.01,                    # Slow decay
    Role.OBSERVATION: 0.05,
    Role.GOAL: 0.02,
    Role.CONSTRAINT: 0.01,
    Role.DECISION: 0.03,
    Role.CONDITIONAL_DEPENDENCY: 0.08,  # Fast decay
}
```

**Operations:**
- `decay_node()` - Reduce activation
- `prune_weak_edges()` - Remove edges < threshold
- `get_inactive_nodes()` - Find nodes to potentially remove

### 7. Conflict Detector

**File:** [cog_memory/conflict_detector.py](../cog_memory/conflict_detector.py)

Identifies contradictions between nodes:

**Detection Methods:**
1. **Negation Detection:** Check for negation words (not, no, never)
2. **Semantic Conflict:** Similar content with opposite negation
3. **Role Conflict:** Constraint vs Goal with similar content

```python
def check_conflict(node1, node2, similarity):
    if similarity < threshold: return False
    has_neg1 = has_negation(node1.text)
    has_neg2 = has_negation(node2.text)
    if has_neg1 != has_neg2:
        return check_semantic_conflict(node1, node2)
    return False
```

## Data Flow

### Ingestion Flow

```
┌─────────────┐
│ Text Input  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ LLMExtractor.extract_commitments()  │ → List[Node]
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ EmbeddingManager.generate_batch()   │ → List[vector]
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ DeduplicationEngine.process_node()  │
│  ├─ query_similar()                 │
│  ├─ should_merge()                  │
│  └─ merge_nodes() or add_new        │
└──────┬──────────────────────────────┘
       │
       ├─────────────────────┬────────────────────┐
       ▼                     ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ LanceStore   │    │ CognitiveGraph│   │ Update edges │
│ .add_node()  │    │ .add_node()  │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

### Query Flow

```
┌─────────────┐
│ Query Text  │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ EmbeddingManager.generate()         │ → vector
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ LanceStore.query_similar()          │ → List[record]
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ CognitiveGraph.activate_node()      │ (for similar)
└──────┬──────────────────────────────┘
       │
       ├─────────────┬──────────────────┐
       ▼            ▼                  ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ activate_goals│ │ propagate()  │ │Combine Scores│
│  (optional)  │ │  (BFS+Stable)│ │ activation + │
└──────────────┘ └──────────────┘ │ similarity   │
                               └──────┬───────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │ List[Node]   │
                               │  (results)   │
                               └──────────────┘
```

**Combined Scoring:**
```python
# Mitigates top-K poisoning from early hops
final_score = (
    activation_weight * node.activation +
    similarity_weight * cosine_similarity(query, node.embedding)
)
```

## LanceDB Schema

**Table:** `nodes`

| Column | Type | Description |
|--------|------|-------------|
| `id` | string | Unique node identifier |
| `text` | string | Node text content |
| `role` | string | Meta-role (fact, goal, etc.) |
| `confidence` | float | Confidence score (0-1) |
| `activation` | float | Current activation level |
| `neighbors` | string (JSON) | Neighbor dict: {id: weight} |
| `metadata` | string (JSON) | Optional metadata |
| `vector` | float[] | Embedding vector |

**Index:** HNSW on `vector` column for similarity search

## Memory Efficiency

### Sparse Representation

- **Neighbors:** Only store explicit connections (not all-pairs)
- **Embeddings:** Persisted to disk (LanceDB), not in memory
- **Graph:** Only node objects and adjacency lists in RAM

### Trade-offs

```
In-Memory: Node objects + adjacency lists (~1KB per node)
On-Disk: Embeddings + full metadata (LanceDB)

For 10,000 nodes:
- RAM: ~10MB (graph) + embeddings cache (optional)
- Disk: ~50-100MB (LanceDB with embeddings)
```

## Activation Propagation Algorithm

**Improved BFS with Active Frontier, Per-Edge Decay, and Stability-Based Stopping:**

```python
def propagate_activation(
    start_nodes: list[Node],
    max_hops: int = 5,
    min_delta: float = 0.01,
    decay_per_hop: float = 0.7,
    stability_epsilon: float = 0.001
) -> bool:
    """
    Returns True if stabilized before max_hops

    Key improvements:
    - Active frontier: Only propagate from nodes above min_delta
    - Per-edge decay: Decay applied at each hop (path-based)
    - Stability check: Stop when max delta < epsilon
    - Edge tracking: Track (parent, child) edges to prevent cycles
    - Multi-parent sum: Contributions from multiple parents are summed
    """
    queue = deque()
    visited_edges = set()

    # Initialize queue with start nodes
    for node in start_nodes:
        queue.append((node.id, 0, None))  # (node_id, hop, parent_id)

    for hop in range(max_hops):
        # Record current activations for stability check
        current_activations = {
            n.id: n.activation for n in graph.get_active_nodes()
            if n.activation > min_delta
        }

        # Process all nodes at current hop
        for _ in range(len(queue)):
            node_id, current_hop, parent_id = queue.popleft()

            node = graph.get_node(node_id)

            # Active frontier: skip weakly activated nodes
            if node.activation < min_delta:
                continue

            # Propagate to neighbors
            for neighbor_id, weight in node.neighbors.items():
                edge = (node_id, neighbor_id)
                if edge in visited_edges:
                    continue  # Prevent cycles

                visited_edges.add(edge)
                neighbor = graph.get_node(neighbor_id)

                # Role-based modulation
                boost = ROLE_BOOSTS.get(
                    (node.role, neighbor.role),
                    1.0  # default
                )

                # Per-edge decay (path-based)
                decay_factor = decay_per_hop ** current_hop
                delta = node.activation * weight * boost * decay_factor

                # Sum contributions from multiple parents
                neighbor.activation += delta

                queue.append((neighbor_id, current_hop + 1, node_id))

        # Check stability: stop if max delta < epsilon
        max_delta = 0
        for node_id, old_activation in current_activations.items():
            new_activation = graph.get_node(node_id).activation
            delta = abs(new_activation - old_activation)
            max_delta = max(max_delta, delta)

        if max_delta < stability_epsilon:
            return True  # System stabilized

    return False  # Reached max_hops
```

**Key Concepts:**
- **Stability vs Activation**: Stability tracks changes between hops; activation determines relevance
- **Active Frontier**: Only nodes above `min_delta` can propagate (efficiency + control)
- **Per-Edge Decay**: Prevents runaway loops and naturally prioritizes near nodes
- **Edge-Based Tracking**: Allows multi-parent nodes while preventing cycles

**Design Note: SUM vs MAX for Multiple Parents**

When a node receives activation from multiple parents, the current implementation uses **SUM accumulation**:
```python
neighbor.activation += delta  # Sums contributions from all parents
```

**Alternative approach (from theory):** Use **MAX** to preserve only the strongest signal:
```python
neighbor.activation = max(neighbor.activation, delta)
```

**Trade-offs:**
- **SUM**: Reinforces nodes that receive multiple weak signals (more inclusive)
- **MAX**: Preserves only the strongest path (prevents overcounting)

Current implementation uses SUM for more inclusive retrieval, but this can be tuned based on use case.

**Complexity:**
- Time: O(V + E) where V = visited nodes, E = traversed edges
- Space: O(depth * branching_factor) for queue

> **See [PROPAGATION_PLAN.md](PROPAGATION_PLAN.md)** for complete implementation roadmap including:
> - Directed edges with optional backward propagation
> - Combined activation + similarity scoring for top-K
> - Two-phase retrieval for mitigating early-hop dominance
> - Parameter reference and tuning guide

## Usage Patterns

### Goal-Driven Reasoning

```python
# Activate specific goals
goals = memory.get_goals()
goal_ids = [g.id for g in goals if "launch" in g.text]

# Query with goal activation
results = memory.query(
    query_text="What affects launch?",
    activate_goals=goal_ids,
)

# Results sorted by activation from propagated influence
```

### Context Management

```python
# Ingest conversation history
for message in conversation:
    memory.ingest_paragraph(message)

# Query for relevant context
context = memory.query(
    query_text="What were the decisions?",
    top_k=5,
)

# Decay old information periodically
memory.apply_decay()
```

## Future Extensions

### Propagation Improvements (See [PROPAGATION_PLAN.md](PROPAGATION_PLAN.md))

1. **Stability-Based Stopping**: Stop propagation when activation stabilizes (delta < epsilon)
2. **Per-Edge Decay**: Path-based decay applied at each hop
3. **Active Frontier**: Only propagate from nodes above threshold
4. **Directed Edges**: Encode causal relationships with optional backward propagation
5. **Combined Scoring**: Weighted mix of activation + similarity for top-K retrieval
6. **Two-Phase Retrieval**: BFS candidate identification + similarity refinement

### System Extensions

1. **Temporal Indexing:** Track timestamps for time-based decay
2. **Conflict Resolution:** Automated merging of conflicting nodes
3. **Hierarchical Goals:** Goal/subgoal relationships
4. **Attention Mechanisms:** Learn edge weights from usage
5. **Multi-Modal:** Support for images, code, structured data
6. **Distributed:** Sharding across multiple LanceDB instances
