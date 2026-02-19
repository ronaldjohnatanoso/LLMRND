# CogMemory: A Brain-Like Memory System for LLMs

## The Problem It Solves

**LLMs have no memory.** They forget everything between conversations.

**CogMemory gives LLMs a brain:**
- Remembers facts across conversations
- Connects related ideas (like your brain does)
- Retrieves relevant context when you ask
- Learns without retraining

---

## How It Works (The Big Picture)

```
┌─────────────────────────────────────────────────────────────┐
│                    COGMEMORY ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────┘

    YOUR TEXT
        │
        ▼
   ┌─────────┐
   │   LLM   │  ← Extracts "commitments" (facts, goals, etc)
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │  Embed  │  ← Converts text to numbers (vectors)
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │  Store  │  ← Saves to database (LanceDB)
   └────┬────┘
        │
        ▼
   ┌─────────────────────────────────────┐
   │          COGNITIVE GRAPH            │
   │                                     │
   │   [Fact] ──── [Goal] ──── [Fact]   │
   │      │           │           │      │
   │      └─────── [Decision] ────┘      │
   │                                     │
   │   Activation spreads like ripples   │
   └─────────────────────────────────────┘
        │
        ▼
    YOUR QUERY
        │
        ▼
   ┌─────────┐
   │  Search │  ← Find similar nodes
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │ Spread  │  ← Activation propagates
   └────┬────┘
        │
        ▼
   ┌─────────┐
   │  Results│  ← Return top activated nodes
   └─────────┘
```

---

## Part 1: The Graph (Your "Brain")

### What Is A Node?

A **node** = one piece of information

```python
Node = {
    "text": "The project deadline is Friday",
    "role": "FACT",              # Type: FACT, GOAL, DECISION, etc
    "activation": 0.85,          # How "active"/relevant right now
    "neighbors": {               # Connected nodes
        "deadline_id": 0.92,     # Edge weight = similarity
        "friday_id": 0.78
    }
}
```

### What Are Edges?

**Edges** = connections between related nodes

```
    [Project] ────── [Friday]
         │              │
         │ 0.85         │ 0.72
         │              │
         ▼              ▼
     [Deadline] ──── [Stress]
              0.91

Numbers = connection strength (similarity)
```

**Key insight:** Undirected = flows both ways
- "Project" activates "Deadline"
- "Deadline" also activates "Project"

---

## Part 2: How Query Works (Step-by-Step)

### Step 1: You Ask A Question

```
Query: "What do I need to finish this week?"
```

### Step 2: Find Direct Matches (Layer 1)

```
Query embedding is compared to all nodes

Matches found:
- "deadline is Friday"      similarity: 0.87  ← activates this
- "finish project report"   similarity: 0.82  ← activates this
- "team meeting tomorrow"   similarity: 0.65  ← activates this
```

### Step 3: Spread Activation (Like Ripples)

```
Hop 0 (Direct matches):
  ✓ deadline_is_Friday = 0.87
  ✓ finish_project = 0.82

Hop 1 (Neighbors of direct matches):
  deadline_is_Friday → project (0.87 × 0.85 × decay)
  deadline_is_Friday → stress (0.87 × 0.91 × decay)
  finish_project → report (0.82 × 0.78 × decay)

Hop 2 (Neighbors of neighbors):
  ... continues until depth limit or stable
```

### Step 4: Return Top Results

```
Sorted by activation (highest first):

1. deadline_is_Friday      0.87  (direct match)
2. finish_project          0.82  (direct match)
3. project                 0.52  (propagated)
4. stress                  0.48  (propagated)
5. report                  0.41  (propagated)
```

---

## Part 3: The Formula (How Activation Spreads)

### The Core Formula

```
child_activation += parent_activation × edge_weight × role_boost × decay
```

### Breakdown

| Component | What It Does | Example |
|-----------|-------------|---------|
| `parent_activation` | How strong is the source? | 0.87 |
| `edge_weight` | How similar are they? | 0.85 (cosine similarity) |
| `role_boost` | Does their relationship matter? | 1.5 (goal→decision) |
| `decay` | Distance penalty | 0.7^hop (70% per hop) |

### Full Example

```
Scenario: Query about "deadline"

Layer 1 (direct):
  deadline_node.activation = 0.87

Layer 2 (one hop away):
  project.activation += 0.87 × 0.85 × 1.0 × 0.7
                       = 0.52

  stress.activation += 0.87 × 0.91 × 1.3 × 0.7
                      = 0.72  (higher! FACT→FACT boost)

Layer 3 (two hops away):
  report.activation += 0.52 × 0.78 × 1.0 × 0.7²
                      = 0.20  (weaker due to decay)
```

---

## Part 4: Role Boosts (Why Some Connections Matter More)

### The Boost Matrix

```
From \ To        │ FACT │ GOAL │ DECISION │ CONSTRAINT
─────────────────┼──────┼──────┼──────────┼────────────
FACT             │ 1.3  │ 1.0  │ 1.0      │ 1.0
GOAL             │ 1.0  │ 1.0  │ 1.5      │ -0.8
DECISION         │ 1.0  │ 1.0  │ 1.0      │ 1.0
CONSTRAINT       │ 1.0  │ 1.0  │ -0.8     │ 1.0
```

### What This Means

```
GOAL → DECISION: 1.5× boost
  "Launch product" activates "Set release date" more strongly

CONSTRAINT → DECISION: -0.8× (inhibition)
  "Budget limit" REDUCES activation of "Hire 10 people"

FACT → FACT: 1.3× boost
  Facts reinforce related facts
```

### Why This Matters

**Real-world analogy:**
- Goals drive decisions (positive boost)
- Constraints inhibit bad decisions (negative boost)
- Facts strengthen related facts (reinforcement)

---

## Part 5: Per-Hop Decay (Why Nearby Stuff Wins)

### The Decay Formula

```
decay_factor = DECAY_PER_HOP ^ hop_count

With DECAY_PER_HOP = 0.7:

Hop 0: 1.00      (100% - source)
Hop 1: 0.70      (70% retained)
Hop 2: 0.49      (49% retained)
Hop 3: 0.34      (34% retained)
Hop 4: 0.24      (24% retained)
Hop 5: 0.17      (17% retained)
```

### Why This Works

```
Query: "project deadline"

Direct hit (hop 0):    deadline = 0.87
One hop away:         project  = 0.52
Two hops away:        stress   = 0.34
Three hops away:      anxiety  = 0.17

Result: Nearby nodes dominate, distant nodes fade
```

### Real-World Analogy

Like throwing a stone in water:
- Splash = direct match (loud)
- First ripples = neighbors (still strong)
- Far ripples = barely visible

---

## Part 6: The Two-Gate System (Quality Control)

### Gate 1: Min Signal Strength (min_delta)

**Question:** "Is this signal strong enough to interview?"

```python
if activation_delta < 0.3:  # Too weak
    continue  # Don't activate
```

**What it does:** Filters weak parent→child signals

### Gate 2: Min Node Activation (activation_threshold)

**Question:** "Is this node strong enough to promote?"

```python
if node.activation < 0.5:  # Too weak
    continue  # Don't propagate further
```

**What it does:** Only strong nodes can activate their children

### Example

```
Layer 1 (direct matches):
  deadline = 0.87  ✓ Strong enough to propagate
  meeting  = 0.35  ✗ Too weak (blocked by Gate 2)

Layer 2 (propagation):
  deadline → project (0.87 × 0.85 × 0.7 = 0.52)  ✓ Passes Gate 1
  deadline → panic  (0.87 × 0.60 × 0.7 = 0.37)  ✗ Blocked by Gate 1
```

### Why Two Gates?

**Single gate problem:** Either too much noise OR too few results

**Two gates:** Balanced retrieval
- Gate 1: Filter weak signals
- Gate 2: Filter weak nodes

---

## Part 7: Handling New Information

### The Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    HOW NEW INFO IS ADDED                     │
└─────────────────────────────────────────────────────────────┘

1. New text arrives
   "The project deadline moved to Monday"

2. Extract commitments (LLM)
   Node: {text: "deadline is Monday", role: FACT}

3. Generate embedding
   Vector: [0.23, -0.45, 0.67, ...]

4. Check for duplicates
   Query LanceDB for similar nodes

5. Decision: MERGE or NEW?
   ┌─────────────────────────────────────────────┐
   │ similarity > 0.95?  → MERGE (duplicate)    │
   │ similarity > 0.75?  → UPDATE (related)     │
   │ similarity < 0.75?  → CREATE NEW            │
   └─────────────────────────────────────────────┘

6a. IF MERGE: Blend embeddings
   V_old = normalize((1 - α) × V_old + α × V_new)
   This updates the memory gradually!

6b. IF NEW: Add to graph + create edges
   Connect to similar existing nodes
```

### The Blend Formula (When Merging)

```python
# Alpha = 0.3 (30% new info, 70% old)
V_new = (1 - 0.3) × V_old + 0.3 × V_new
V_new = normalize(V_new)

Result: Memory adapts without changing too fast
```

### Real-World Analogy

Like updating a mental model:
- You hear "deadline is Monday" (new info)
- You already knew "deadline is Friday" (old info)
- Your brain blends them: "deadline might be flexible"

---

## Part 8: Scenarios & Use Cases

### Scenario 1: Project Planning

```
User: "What do I need to do for the project?"

Layer 1 (direct matches):
  - "finish report"        0.87
  - "team meeting"         0.72

Layer 2 (propagated):
  - "deadline"             0.52  (from "finish report")
  - "slides"               0.48  (from "finish report")
  - "prep agenda"          0.41  (from "team meeting")

Result: Returns action items + related context
```

### Scenario 2: Goal Achievement

```
User activates goal: "Launch product"

Propagation (role boosts active):
  Launch → Set release date    ×1.5 boost
  Launch → Marketing plan      ×1.5 boost
  Launch → Budget limit       ×-0.8 inhibition

Result: Goal activates related decisions + inhibits conflicts
```

### Scenario 3: Conflict Detection

```
Ingested: "We have unlimited budget"
Existing: "Budget limit is $10k"

Similarity = 0.82  (same topic, different)
Action: BOTH stored, marked as conflict
Result: Query shows both with "CONFLICT" tag
```

---

## Part 9: Limitations & Gotchas

### What It Does Well

✅ **Associative recall** - "What was related to X?"
✅ **Context retrieval** - "What do I need to know about project?"
✅ **Goal-driven reasoning** - "What follows from this goal?"
✅ **Incremental learning** - Absorbs new info over time

### What It Doesn't Do

❌ **Temporal reasoning** - Doesn't understand time/order
❌ **Causal inference** - Doesn't know "A caused B"
❌ **Quantitative reasoning** - Can't do math
❌ **Perfect accuracy** - May return irrelevant stuff (tune thresholds!)

### Common Gotchas

#### Gotcha 1: Over-Propagation

**Problem:** Activation spreads too far, returns noise

**Symptom:** Query returns unrelated stuff

**Fix:** Increase `min_delta` (Gate 1) or `decay_per_hop`

```
min_delta: 0.3 → 0.4   (stricter filter)
decay_per_hop: 0.8 → 0.7  (more decay)
```

#### Gotcha 2: Under-Propagation

**Problem:** Activation dies too fast

**Symptom:** Only direct matches, no related context

**Fix:** Decrease thresholds

```
min_delta: 0.4 → 0.2   (more permissive)
decay_per_hop: 0.6 → 0.8  (less decay)
```

#### Gotcha 3: Loop Confusion

**Problem:** A → B → A (bidirectional edges)

**Question:** "Is this a bug?"

**Answer:** No! Decay naturally limits loops. Each pass through the loop applies more decay.

```
A → B: 0.87 × 0.7 = 0.61
B → A: 0.61 × 0.7 = 0.43  (weaker!)
A → B: 0.43 × 0.7 = 0.30  (dies out)
```

#### Gotcha 4: Edge Duplication

**Problem:** "Why do I see A→B AND B→A?"

**Answer:** Undirected graph = both directions exist

**Visualization:**
```
    A ⇄ B

Stored as:
  A.neighbors = {B: 0.85}
  B.neighbors = {A: 0.85}
```

This is correct! Enables bidirectional activation flow.

---

## Part 10: Parameter Tuning Guide

### The 5 Critical Parameters

| Parameter | Default | What It Controls | When to Change |
|-----------|---------|------------------|----------------|
| `min_delta` | 0.3 | Min signal to activate child | Too much noise? ↑ it |
| `activation_threshold` | 0.5 | Min activation to propagate | Too few results? ↓ it |
| `decay_per_hop` | 0.7 | How fast activation fades | Too much spread? ↓ it |
| `propagation_depth` | 2 | How many hops to explore | Want more context? ↑ it |
| `min_similarity` | 0.55 | Min similarity for Layer 1 | Too many matches? ↑ it |

### Tuning Scenarios

#### Scenario: "Too much noise, irrelevant results"

```
Change these:
  min_delta: 0.3 → 0.4           (stricter Gate 1)
  min_similarity: 0.55 → 0.65    (stricter Layer 1)
  decay_per_hop: 0.7 → 0.6       (more decay)
```

#### Scenario: "Not enough results, missing context"

```
Change these:
  min_delta: 0.3 → 0.2           (looser Gate 1)
  activation_threshold: 0.5 → 0.4  (looser Gate 2)
  decay_per_hop: 0.7 → 0.8       (less decay)
  propagation_depth: 2 → 3       (explore farther)
```

#### Scenario: "Only direct matches, no propagation"

```
Likely cause: activation_threshold too high

Fix: activation_threshold: 0.5 → 0.3
```

---

## Part 11: Pseudocode (Complete Query Flow)

```python
def query(graph, query_text, top_k=10):
    """
    Complete query flow in simple pseudocode
    """

    # Step 1: Generate query embedding
    query_vector = embed(query_text)

    # Step 2: Find direct matches (Layer 1)
    similar_nodes = graph.search_similar(
        query_vector,
        k=top_k * 2,  # Fetch extra, will filter
        threshold=0.55
    )

    # Step 3: Activate direct matches
    for node in similar_nodes:
        node.activation = node.similarity_to_query
        node.similarity_to_query = node.similarity

    # Step 4: Propagate activation (BFS)
    for start_node in similar_nodes:
        queue = [(start_node, 0)]  # (node, hop_count)
        visited_edges = set()

        while queue:
            node, hop = queue.pop(0)

            # Gate 2: Is node strong enough to propagate?
            if node.activation < 0.5:
                continue

            # Propagate to neighbors
            for neighbor_id, edge_weight in node.neighbors:
                edge = (node.id, neighbor_id)
                if edge in visited_edges:
                    continue  # Prevent cycles
                visited_edges.add(edge)

                neighbor = graph.get(neighbor_id)

                # Calculate role boost
                boost = ROLE_BOOSTS.get((node.role, neighbor.role), 1.0)

                # Calculate decay
                decay = 0.7 ** hop

                # Calculate delta
                delta = (
                    node.similarity_to_query *
                    edge_weight *
                    boost *
                    decay
                )

                # Gate 1: Is signal strong enough?
                if delta < 0.3:
                    continue

                # Activate neighbor
                neighbor.activation += delta

                # Add to queue if not at max depth
                if hop + 1 < 2:  # propagation_depth = 2
                    queue.append((neighbor, hop + 1))

    # Step 5: Return top activated nodes
    results = graph.get_top_activated(top_k)
    return results
```

---

## Part 12: Quick Reference

### File Structure

```
cog_memory/
├── node.py              # Node class (data structure)
├── cognitive_graph.py   # Graph + propagation logic
├── query_interface.py   # Main API (query, ingest)
├── llm_extractor.py     # Extract commitments from text
├── embedding_manager.py # Generate embeddings
├── lance_store.py       # Vector database
├── deduplication.py     # Merge similar nodes
└── decay.py             # Time-based forgetting

docs/
├── GUIDE.md             # THIS FILE (start here!)
├── ARCHITECTURE.md      # Technical details
└── PROPAGATION_PLAN.md  # Implementation roadmap
```

### Key Classes & Methods

```python
# Main interface
memory = CognitiveMemory()

# Add information
memory.ingest_paragraph("The deadline is Friday")

# Query
results = memory.query(
    query_text="What's due?",
    top_k=10,
    propagation_depth=2,
    decay_per_hop=0.7
)

# Each result is a Node:
for node in results:
    print(f"{node.text} (activation: {node.activation})")
```

---

## Summary: The Mental Model

Think of CogMemory as **a brain that stores connected ideas**:

1. **Nodes** = individual ideas/facts
2. **Edges** = connections between related ideas
3. **Activation** = how "relevant" right now
4. **Propagation** = spreading relevance like ripples
5. **Decay** = nearby stuff wins, distant fades
6. **Role boosts** = some connections matter more

**Key insight:** It's not just similarity search - it's **associative recall** via spreading activation, just like your brain does.

---

## Still Confused?

**Common questions:**

Q: "Why not just use vector search?"
A: Vector search finds similar items. CogMemory finds **connected** items (associations).

Q: "What's the difference between Layer 1 and propagated?"
A: Layer 1 = direct matches (similarity). Propagated = connected neighbors (associations).

Q: "Do I need to train anything?"
A: No! Fully query-time, no training needed.

Q: "Can it handle contradictions?"
A: Yes! Both sides stored, can be marked as conflicts.

Q: "How does it learn?"
A: Through the blend formula - gradually updates embeddings when merging similar nodes.
