# CogMemory Code Map

## Project Structure

```
LLMRND/
├── cog_memory/              # Core package
│   ├── __init__.py          # Package exports
│   ├── node.py              # Node data structure and Role enum
│   ├── cognitive_graph.py   # Graph with activation propagation
│   ├── lance_store.py       # LanceDB persistent storage
│   ├── llm_extractor.py     # LLM commitment extraction
│   ├── embedding_manager.py # Embedding generation
│   ├── deduplication.py     # Semantic deduplication
│   ├── conflict_detector.py # Contradiction detection
│   ├── decay.py             # Activation decay mechanism
│   └── query_interface.py   # Main API (CognitiveMemory)
├── tests/                   # Unit tests
├── examples/                # Usage examples
├── docs/                    # Additional documentation
└── README.md                # This file
```

## Component Reference

### Core Data Structures

#### [cog_memory/node.py](cog_memory/node.py)

**Classes:**
- `Role` (Enum) - Meta-roles for commitments
  - `FACT`, `OBSERVATION`, `GOAL`, `CONSTRAINT`, `DECISION`, `CONDITIONAL_DEPENDENCY`

- `Node` (dataclass) - Represents a commitment
  - Fields: `id`, `text`, `role`, `confidence`, `activation`, `neighbors`, `metadata`, `embedding`
  - Methods: `add_neighbor()`, `remove_neighbor()`, `update_activation()`, `reset_activation()`, `to_dict()`, `from_dict()`

### Graph and Activation

#### [cog_memory/cognitive_graph.py](cog_memory/cognitive_graph.py)

**Classes:**
- `CognitiveGraph` - Sparse graph with activation propagation
  - Methods:
    - `add_node(node)` - Add node to graph
    - `get_node(node_id)` - Retrieve node by ID
    - `remove_node(node_id)` - Remove node and its edges
    - `add_edge(source_id, target_id, weight)` - Add directed edge
    - `activate_node(node_id, activation)` - Set node activation
    - `propagate_activation(start_node_id, depth)` - BFS activation propagation
    - `activate_goals(goal_ids, activation)` - Activate multiple goals
    - `reset_all_activations()` - Zero all activations
    - `get_top_activated(n)` - Get N most activated nodes
    - `get_nodes_by_role(role)` - Filter nodes by role

**Constants:**
- `ROLE_BOOSTS` - Role-to-role activation multipliers

### Storage

#### [cog_memory/lance_store.py](cog_memory/lance_store.py)

**Classes:**
- `NodeRecord` (Pydantic) - LanceDB table schema
- `LanceStore` - Persistent vector storage wrapper
  - Methods:
    - `add_node(...)` - Insert node with embedding
    - `query_similar(embedding, k, filter_role, min_confidence)` - Similarity search
    - `get_node(node_id)` - Retrieve node by ID
    - `update_node(node_id, updates)` - Update node fields
    - `delete_node(node_id)` - Remove node from database
    - `get_all_nodes()` - Retrieve all nodes
    - `count_nodes()` - Get total node count

### Extraction and Embedding

#### [cog_memory/llm_extractor.py](cog_memory/llm_extractor.py)

**Classes:**
- `LLMExtractor` - LLM-based commitment extraction
  - Methods:
    - `extract_commitments(text)` - Returns list of Nodes
    - `_dummy_extract(text)` - Fallback rule-based extraction

#### [cog_memory/embedding_manager.py](cog_memory/embedding_manager.py)

**Classes:**
- `EmbeddingManager` - Embedding generation
  - Methods:
    - `generate_embedding(text)` - Single text embedding
    - `generate_embeddings_batch(texts)` - Batch embedding

### Deduplication and Conflict

#### [cog_memory/deduplication.py](cog_memory/deduplication.py)

**Classes:**
- `DeduplicationEngine` - Semantic node merging
  - Methods:
    - `find_similar(node, embedding, store)` - Find similar nodes
    - `should_merge(node, similar_node)` - Merge decision
    - `merge_nodes(existing, new)` - Merge two nodes
    - `process_node(node, embedding, store)` - Full deduplication pipeline

#### [cog_memory/conflict_detector.py](cog_memory/conflict_detector.py)

**Classes:**
- `ConflictDetector` - Contradiction detection
  - Methods:
    - `has_negation(text)` - Check for negation words
    - `check_conflict(node1, node2, similarity)` - Detect conflicts
    - `find_conflicts(node, candidates, similarities)` - Find all conflicts

### Maintenance

#### [cog_memory/decay.py](cog_memory/decay.py)

**Classes:**
- `DecayModule` - Activation decay and edge pruning
  - Methods:
    - `decay_node(node)` - Apply decay to single node
    - `decay_graph(nodes)` - Apply decay to all nodes
    - `prune_weak_edges(nodes)` - Remove weak connections
    - `get_inactive_nodes(nodes)` - Find inactive nodes
    - `boost_node(node, amount)` - Increase activation

**Constants:**
- `ROLE_DECAY_RATES` - Decay rates per role

### Main API

#### [cog_memory/query_interface.py](cog_memory/query_interface.py)

**Classes:**
- `CognitiveMemory` - Main interface orchestrating all components
  - Methods:
    - `ingest_paragraph(text)` - Extract, embed, store commitments
    - `query(query_text, top_k, activate_goals)` - Query with activation
    - `get_nodes_by_role(role)` - Filter by role
    - `get_goals()` - Get all goal nodes
    - `get_constraints()` - Get all constraint nodes
    - `apply_decay()` - Apply decay and pruning
    - `get_stats()` - System statistics

## Meta-Role Activation Rules

| Source Role | Target Role | Effect | Multiplier |
|-------------|-------------|--------|------------|
| Goal | Decision | Excitatory | 1.5 |
| Constraint | Decision | Inhibitory | -0.8 |
| Observation | Decision | Supporting | 1.2 |
| Fact | Fact | Reinforce | 1.3 |
| ConditionalDependency | Decision | Bridge | 1.1 |
| Goal | ConditionalDependency | Enable | 1.2 |

## Data Flow

### Ingestion Pipeline

```
Text Input
    ↓
LLMExtractor.extract_commitments()
    ↓
List[Node] with meta-roles
    ↓
EmbeddingManager.generate_embeddings_batch()
    ↓
List[List[float]] (embeddings)
    ↓
DeduplicationEngine.process_node()
    ↓
LanceStore.add_node() OR merge with existing
    ↓
CognitiveGraph.add_node()
    ↓
Update edges and neighbors
```

### Query Pipeline

```
Query Text
    ↓
EmbeddingManager.generate_embedding()
    ↓
LanceStore.query_similar()
    ↓
CognitiveGraph.activate_node() (for similar nodes)
    ↓
CognitiveGraph.activate_goals() (if specified)
    ↓
CognitiveGraph.propagate_activation()
    ↓
CognitiveGraph.get_top_activated()
    ↓
List[Node] sorted by activation
```

## Key Patterns

### Node Creation

```python
from cog_memory import Node, Role

node = Node(
    text="Launch product by Q3",
    role=Role.GOAL,
    confidence=0.9,
)
```

### Memory Initialization

```python
from cog_memory import CognitiveMemory

# With OpenAI
memory = CognitiveMemory()

# With dummy extractor (testing)
memory = CognitiveMemory(use_dummy_extractor=True)

# With sentence-transformers
memory = CognitiveMemory(use_sentence_transformer=True)
```

### Activation Propagation

```python
# Manual activation
memory.graph.activate_node(node_id, 1.0)
memory.graph.propagate_activation(node_id, depth=2)

# Goal-driven
results = memory.query(
    query_text="...",
    activate_goals=[goal_id1, goal_id2],
)
```
