# CogMemory: LLM Context Compression & Cognitive Graph System

An intelligent context management system for LLM applications that extracts commitments from long-form text, assigns meta-roles, stores embeddings persistently, and maintains a sparse cognitive graph with activation propagation for goal-driven reasoning.

## Overview

CogMemory addresses the problem of context window limitations in LLMs by:

- Extracting semantic commitments from text with meta-role classification
- Storing embeddings in LanceDB for persistent, RAM-efficient storage
- Maintaining a sparse cognitive graph with activation propagation
- Supporting goal-driven retrieval with multi-hop reasoning
- Implementing deduplication and conflict detection

## Installation

### Requirements

- Python 3.10+
- OpenAI API key (or use sentence-transformers for local embeddings)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LLMRND.git
cd LLMRND
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Quick Start

```python
from cog_memory import CognitiveMemory

# Initialize the system
memory = CognitiveMemory(
    db_path="./data/lancedb",
    use_dummy_extractor=False,  # Set True for testing without API
)

# Ingest text
text = """
We need to launch the product by Q3. The budget is limited to $50k.
Marketing should start 2 weeks before launch. The goal is 1000 users by end of year.
"""

nodes = memory.ingest_paragraph(text)
print(f"Created {len(nodes)} nodes")

# Query with goal activation
results = memory.query(
    query_text="What are the launch constraints?",
    top_k=5,
)

for node in results:
    print(f"{node.role.value}: {node.text} (activation: {node.activation:.2f})")
```

## Meta-Roles

Each commitment is assigned a meta-role that determines its behavior in the cognitive graph:

| Role | Description | Example |
|------|-------------|---------|
| `fact` | Verifiable information | "Python is an interpreted language" |
| `observation` | Noted information | "Users seem to prefer dark mode" |
| `goal` | Target state or objective | "Achieve 1000 active users" |
| `constraint` | Limitation or restriction | "Budget limited to $50k" |
| `decision` | Chosen course of action | "Launch on Monday" |
| `conditional_dependency` | If-then relationship | "If tests pass, then deploy" |

## Architecture

### Core Components

- **[Node](cog_memory/node.py)** - Data structure representing commitments
- **[CognitiveGraph](cog_memory/cognitive_graph.py)** - Graph with activation propagation
- **[LanceStore](cog_memory/lance_store.py)** - Persistent vector storage
- **[LLMExtractor](cog_memory/llm_extractor.py)** - LLM-based commitment extraction
- **[EmbeddingManager](cog_memory/embedding_manager.py)** - Embedding generation
- **[DeduplicationEngine](cog_memory/deduplication.py)** - Semantic deduplication
- **[ConflictDetector](cog_memory/conflict_detector.py)** - Contradiction detection
- **[DecayModule](cog_memory/decay.py)** - Activation decay and forgetting
- **[CognitiveMemory](cog_memory/query_interface.py)** - Main API interface

### Data Flow

```
Text Input
    ↓
LLM Extractor → Commitments with meta-roles
    ↓
Embedding Manager → Vector embeddings
    ↓
Deduplication Engine → Merge/link similar nodes
    ↓
LanceDB + Cognitive Graph → Persistent storage
    ↓
Query Interface → Goal-driven activation → Results
```

## Usage Examples

### Basic Query

```python
from cog_memory import CognitiveMemory

memory = CognitiveMemory()

# Query for relevant context
results = memory.query(
    query_text="What are the project goals?",
    top_k=10,
)
```

### Goal-Driven Retrieval

```python
# Get all goals
goals = memory.get_goals()

# Activate specific goals and propagate
goal_ids = [g.id for g in goals[:3]]
results = memory.query(
    query_text="What decisions relate to these goals?",
    activate_goals=goal_ids,
)
```

### Apply Decay

```python
# Decay activations and prune weak edges
stats = memory.apply_decay()
print(f"Decayed {stats['nodes_decayed']} nodes")
print(f"Pruned {stats['edges_pruned']} edges")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Using Dummy Extractor (No API Required)

```python
memory = CognitiveMemory(
    use_dummy_extractor=True,  # Uses rule-based extraction
)
```

### Using Sentence Transformers (Local Embeddings)

```python
memory = CognitiveMemory(
    use_sentence_transformer=True,
)
```

## Configuration

Environment variables (see [.env.example](.env.example)):

- `OPENAI_API_KEY` - OpenAI API key
- `OPENAI_MODEL` - Model for extraction (default: gpt-4o)
- `EMBEDDING_MODEL` - Embedding model (default: text-embedding-3-small)
- `LANCEDB_PATH` - Path to LanceDB storage (default: ./data/lancedb)

## Documentation

- [Architecture Details](docs/ARCHITECTURE.md) - Deep dive into system design
- [Code Map](CODEMAP.md) - Component reference and data flow

## License

MIT License - see LICENSE file for details
