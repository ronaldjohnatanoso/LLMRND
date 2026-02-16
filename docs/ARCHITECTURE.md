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

**Activation Propagation (BFS):**
```
propagate_activation(start_node, depth=2):
    queue = [(start_node, 0)]
    while queue:
        node, current_depth = queue.pop()
        if current_depth >= depth: continue
        for neighbor_id, weight in node.neighbors:
            role_boost = ROLE_BOOSTS[(node.role, neighbor.role)]
            neighbor.activation += node.activation * weight * role_boost
            queue.append((neighbor, current_depth + 1))
```

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
│ activate_goals│ │ propagate()  │ │get_top_active│
│  (optional)  │ │  (BFS)       │ │   (sorted)   │
└──────────────┘ └──────────────┘ └──────┬───────┘
                                             │
                                             ▼
                                      ┌──────────────┐
                                      │ List[Node]   │
                                      │  (results)   │
                                      └──────────────┘
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

**Breadth-First Search with Role Modulation:**

```python
def propagate_activation(start_node, depth=2):
    queue = deque([(start_node.id, 0)])
    visited = set()

    while queue:
        node_id, current_depth = queue.popleft()

        if current_depth >= depth or node_id in visited:
            continue

        visited.add(node_id)
        node = graph.get_node(node_id)

        if node.activation < threshold:
            continue

        for neighbor_id, weight in node.neighbors:
            neighbor = graph.get_node(neighbor_id)

            # Role-based modulation
            boost = ROLE_BOOSTS.get(
                (node.role, neighbor.role),
                default_boost
            )

            # Apply activation
            delta = node.activation * weight * boost
            neighbor.update_activation(delta)

            # Continue propagation
            if current_depth + 1 < depth:
                queue.append((neighbor_id, current_depth + 1))
```

**Complexity:**
- Time: O(V + E) where V = visited nodes, E = traversed edges
- Space: O(depth * branching_factor) for queue

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

1. **Temporal Indexing:** Track timestamps for time-based decay
2. **Conflict Resolution:** Automated merging of conflicting nodes
3. **Hierarchical Goals:** Goal/subgoal relationships
4. **Attention Mechanisms:** Learn edge weights from usage
5. **Multi-Modal:** Support for images, code, structured data
6. **Distributed:** Sharding across multiple LanceDB instances
