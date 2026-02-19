# CogMemory Visual Guide: Diagrams & Examples

## Diagram 1: The Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TEXT TO MEMORY PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

   User Input
   "The project deadline is Friday and we need to finish the report"
        │
        ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                     LLM EXTRACTOR                                │
   │                                                                  │
   │  Extracts "commitments" (meaningful units):                      │
   │  • "project deadline is Friday"         [FACT]                  │
   │  • "finish the report"                  [DECISION]               │
   │                                                                  │
   │  Output: List[Node]                                              │
   └────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                   EMBEDDING MANAGER                              │
   │                                                                  │
   │  Converts text to vectors (numbers):                             │
   │  "project deadline is Friday" → [0.23, -0.45, 0.67, ...]       │
   │  "finish the report"         → [0.11, 0.33, -0.21, ...]        │
   │                                                                  │
   │  Output: List[Vector]                                            │
   └────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                   SEMANTIC EDGE CREATION                         │
   │                                                                  │
   │  Compares embeddings, creates edges if similar:                 │
   │                                                                  │
   │  [deadline] ───────────── [report]                             │
   │        similarity: 0.82 (edge weight)                           │
   │                                                                  │
   │  Both nodes get each other in neighbors:                        │
   │  deadline.neighbors = {report: 0.82}                           │
   │  report.neighbors = {deadline: 0.82}                           │
   │                                                                  │
   └────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                   DEDUPLICATION CHECK                           │
   │                                                                  │
   │  Is this new? Or similar to existing?                           │
   │                                                                  │
   │  similarity > 0.95? → MERGE (blend embeddings)                  │
   │  similarity > 0.75? → UPDATE (increase confidence)              │
   │  similarity < 0.75? → CREATE NEW NODE                           │
   │                                                                  │
   └────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                    LANCEDB (Database)                           │
   │                                                                  │
   │  Stores:                                                         │
   │  • Node metadata                                                 │
   │  • Embedding vector                                              │
   │  • Neighbor connections                                          │
   │                                                                  │
   └────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                   COGNITIVE GRAPH (Memory)                      │
   │                                                                  │
   │     [deadline] ←────────→ [report]                              │
   │         ↓                     ↓                                  │
   │     [project] ←────────→ [finish]                               │
   │                                                                  │
   │  In-memory structure for fast traversal                         │
   └──────────────────────────────────────────────────────────────────┘
```

---

## Diagram 2: Query Flow - Step by Step

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         QUERY EXECUTION                                │
└─────────────────────────────────────────────────────────────────────────┘

   User Query: "What do I need to do this week?"
        │
        ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                  GENERATE QUERY EMBEDDING                        │
   │                                                                  │
   │  "What do I need to do this week?"                              │
   │  → [0.45, 0.23, -0.11, ...]                                     │
   └────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                   LAYER 1: DIRECT MATCHES                       │
   │                                                                  │
   │  Search all nodes by similarity:                                │
   │                                                                  │
   │  ┌─────────────────────────────────────────────────────────┐   │
   │  │ "finish project report"    similarity: 0.87  ✓ ACTIVATE  │   │
   │  │ "deadline is Friday"        similarity: 0.82  ✓ ACTIVATE  │   │
   │  │ "team meeting tomorrow"     similarity: 0.65  ✓ ACTIVATE  │   │
   │  │ "coffee shop nearby"         similarity: 0.42  ✗ reject    │   │
   │  └─────────────────────────────────────────────────────────┘   │
   │                                                                  │
   │  Set node.activation = node.similarity_to_query                │
   └────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                   LAYER 2: PROPAGATION (Hop 1)                  │
   │                                                                  │
   │  From each Layer 1 node, activate neighbors:                   │
   │                                                                  │
   │  From "finish report" (activation=0.87):                        │
   │    → "project"    : 0.87 × 0.85 × 1.0 × 0.7 = 0.52 ✓          │
   │    → "deadline"   : 0.87 × 0.82 × 1.0 × 0.7 = 0.50 ✓          │
   │    → "slides"     : 0.87 × 0.45 × 1.0 × 0.7 = 0.28 ✗ (Gate 1) │
   │                                                                  │
   │  From "deadline is Friday" (activation=0.82):                   │
   │    → "project"    : 0.82 × 0.91 × 1.0 × 0.7 = 0.52 ✓          │
   │    → "stress"     : 0.82 × 0.73 × 1.3 × 0.7 = 0.55 ✓          │
   │                                                                  │
   │  Note: "project" gets activated TWICE, sums to ~1.04           │
   │        (clamped to max 1.0)                                     │
   └────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                   LAYER 3: PROPAGATION (Hop 2)                  │
   │                                                                  │
   │  From Layer 2 nodes, activate THEIR neighbors:                 │
   │                                                                  │
   │  From "project" (activation=1.0, maxed out):                   │
   │    → "launch"     : 1.0 × 0.65 × 1.5 × 0.7² = 0.48 ✓         │
   │    → "team"       : 1.0 × 0.52 × 1.0 × 0.7² = 0.26 ✗ (Gate 1)│
   │                                                                  │
   │  From "stress" (activation=0.55):                               │
   │    → "anxiety"    : 0.55 × 0.81 × 1.0 × 0.7² = 0.22 ✗        │
   │                                                                  │
   │  From "deadline" (activation=0.50):                             │
   │    → "reminder"   : 0.50 × 0.44 × 1.0 × 0.7² = 0.11 ✗        │
   │                                                                  │
   └────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                   FINAL RESULTS (Top 5)                         │
   │                                                                  │
   │  Rank  Node                    Activation   Source              │
   │  ─────────────────────────────────────────────────────────────  │
   │   1    finish report           0.87       Layer 1 (direct)     │
   │   2    deadline is Friday      0.82       Layer 1 (direct)     │
   │   3    project                 1.00       Layer 2 (propagated) │
   │   4    stress                  0.55       Layer 2 (propagated) │
   │   5    launch                  0.48       Layer 3 (propagated) │
   │                                                                  │
   └──────────────────────────────────────────────────────────────────┘
```

---

## Diagram 3: The Two-Gate System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     QUALITY CONTROL: TWO GATES                         │
└─────────────────────────────────────────────────────────────────────────┘

   Incoming signal from parent node
              │
              ▼
       ┌─────────────┐
       │   GATE 1    │
       │  min_delta  │
       │  (default:  │
       │    0.3)     │
       └─────┬───────┘
             │
     ┌───────┴────────┐
     │                │
  PASSED          FAILED
 (delta≥0.3)     (delta<0.3)
     │                │
     │                ▼
     │          Don't activate
     │          (signal too weak)
     │
     ▼
┌─────────────────────────────┐
│   ACTIVATE CHILD NODE       │
│                             │
│   child.activation += delta │
└──────────────┬──────────────┘
               │
               ▼
       ┌─────────────┐
       │   GATE 2    │
       │ activation_ │
       │ threshold   │
       │ (default:   │
       │    0.5)     │
       └─────┬───────┘
             │
     ┌───────┴────────┐
     │                │
  PASSED          FAILED
 (activ≥0.5)     (activ<0.5)
     │                │
     │                ▼
     │          Stop here
     │          (too weak to
     │           propagate)
     │
     ▼
┌─────────────────────────────┐
│   ADD TO QUEUE              │
│   (continue propagating)    │
└─────────────────────────────┘


EXAMPLE GATE 1 FILTER:
─────────────────────────────

Parent: "deadline" (activation=0.87)
Child:  "coffee_shop" (edge_weight=0.3)

Delta = 0.87 × 0.3 × 1.0 × 0.7
      = 0.18

Gate 1: 0.18 < 0.3? ✗ FAILED

Result: "coffee_shop" NOT activated


EXAMPLE GATE 2 FILTER:
─────────────────────────────

Parent: "deadline" → Child: "project"
Delta = 0.52 ✓ (passed Gate 1)
project.activation = 0.52

Gate 2: 0.52 ≥ 0.5? ✓ PASSED

Result: "project" CAN propagate to its children
```

---

## Diagram 4: Role Boost Matrix (Visual)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     ROLE BOOST MATRIX                                 │
│                  How different nodes interact                         │
└─────────────────────────────────────────────────────────────────────────┘

                    TO (TARGET NODE)
                    ┌────────┬────────┬───────────┬────────────┐
                    │  FACT  │  GOAL  │ DECISION  │ CONSTRAINT │
        ┌───────────┼────────┼────────┼───────────┼────────────┤
   F    │   FACT    │  1.3×  │  1.0×  │   1.0×    │    1.0×    │
   R    ├───────────┼────────┼────────┼───────────┼────────────┤
   O    │   GOAL    │  1.0×  │  1.0×  │   1.5×    │   -0.8×    │
   M    ├───────────┼────────┼────────┼───────────┼────────────┤
        │ DECISION  │  1.0×  │  1.0×  │   1.0×    │   -0.8×    │
        ├───────────┼────────┼────────┼───────────┼────────────┤
        │CONSTRAINT │  1.0×  │  1.0×  │  -0.8×    │    1.0×    │
        └───────────┴────────┴────────┴───────────┴────────────┘


WHAT THIS MEANS:
──────────────────────────────────────────────────────────────────────────

1. FACT → FACT (1.3×): Reinforcement
   "It's raining" + "I need an umbrella" = STRONGER association

2. GOAL → DECISION (1.5×): Goals drive decisions
   Goal: "Launch product" → activates "Set release date" MORE

3. CONSTRAINT → DECISION (-0.8×): Constraints inhibit
   Constraint: "Budget limit" → REDUCES "Hire 10 people"

4. DECISION → CONSTRAINT (-0.8×): Decisions respect constraints
   Decision: "Hire team" → REDUCES "Work alone"


EXAMPLE WITH BOOSTS:
──────────────────────────────────────────────────────────────────────────

Query: "launch product"

Layer 1 (direct):
  "launch product" (GOAL) = 0.87

Layer 2 (with role boosts):
  → "set release date" (DECISION)
     Base: 0.87 × 0.85 × 0.7 = 0.52
     With GOAL→DECISION boost (1.5×): 0.52 × 1.5 = 0.78 ✓

  → "budget limit" (CONSTRAINT)
     Base: 0.87 × 0.65 × 0.7 = 0.40
     With GOAL→CONSTRAINT boost (1.0×): 0.40 × 1.0 = 0.40

  → "hire team" (DECISION)
     Base: 0.87 × 0.72 × 0.7 = 0.44
     With GOAL→DECISION boost (1.5×): 0.44 × 1.5 = 0.66 ✓

  → "reduce cost" (DECISION)
     Base: 0.87 × 0.68 × 0.7 = 0.41
     But "budget limit" (CONSTRAINT) → "hire team" (DECISION) = -0.8×
     Final: 0.41 × -0.8 = -0.33 ✗ (inhibited!)
```

---

## Diagram 5: Per-Hop Decay Visual

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PER-HOP DECAY IN ACTION                            │
└─────────────────────────────────────────────────────────────────────────┘

Query: "project deadline"

Layer 0 (Direct Match):
┌──────────────────┐
│  deadline_is_Friday
│  activation: 0.87
│  (100% retained)
└─────────┬────────┘
          │
          │ decay = 0.7^1 = 0.7
          ▼
Layer 1 (One Hop Away):
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  project         │  │  stress          │  │  team            │
│  activ: 0.52     │  │  activ: 0.55     │  │  activ: 0.48     │
│  (60% of 0.87)   │  │  (63% of 0.87)   │  │  (55% of 0.87)   │
└────┬─────────────┘  └──────────────────┘  └──────────────────┘
     │
     │ decay = 0.7^2 = 0.49
     ▼
Layer 2 (Two Hops Away):
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  launch          │  │  report          │  │  meeting         │
│  activ: 0.35     │  │  activ: 0.31     │  │  activ: 0.28     │
│  (40% of 0.87)   │  │  (36% of 0.87)   │  │  (32% of 0.87)   │
└────┬─────────────┘  └──────────────────┘  └──────────────────┘
     │
     │ decay = 0.7^3 = 0.34
     ▼
Layer 3 (Three Hops Away):
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  marketing       │  │  slides          │  │  prep            │
│  activ: 0.22     │  │  activ: 0.19     │  │  activ: 0.17     │
│  (25% of 0.87)   │  │  (22% of 0.87)   │  │  (20% of 0.87)   │
└──────────────────┘  └──────────────────┘  └──────────────────┘

PATTERN: Activation decreases exponentially with distance


DECAY COMPARISON:
──────────────────────────────────────────────────────────────────────────

decay_per_hop = 0.5 (aggressive):
  Hop 0: 1.00
  Hop 1: 0.50
  Hop 2: 0.25
  Hop 3: 0.13
  Hop 4: 0.06
  Hop 5: 0.03

decay_per_hop = 0.7 (default):
  Hop 0: 1.00
  Hop 1: 0.70
  Hop 2: 0.49
  Hop 3: 0.34
  Hop 4: 0.24
  Hop 5: 0.17

decay_per_hop = 0.9 (minimal):
  Hop 0: 1.00
  Hop 1: 0.90
  Hop 2: 0.81
  Hop 3: 0.73
  Hop 4: 0.66
  Hop 5: 0.59

TRADEOFF:
• Lower decay = more focused results (nearby wins)
• Higher decay = broader results (farther included)
```

---

## Diagram 6: Merge vs New Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     INGESTION: MERGE OR NEW?                          │
└─────────────────────────────────────────────────────────────────────────┘

   New information arrives: "The deadline moved to Monday"
        │
        ▼
   Generate embedding
        │
        ▼
   Search existing nodes for similarity
        │
        ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │                    FIND MOST SIMILAR                             │
   │                                                                  │
   │  Found: "deadline is Friday" (similarity: 0.89)                 │
   └────────────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │  similarity > 0.95?          │
                        └────┬───────────────────────┬─┘
                             │ YES                   │ NO
                             ▼                       ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │     MERGE       │     │  similarity >   │
                    │   (Duplicate)   │     │     0.75?       │
                    └────────┬────────┘     └────┬────────────┘
                             │                   │ YES    │ NO
                             ▼                   ▼       ▼
                    ┌─────────────────┐     ┌──────────┐ ┌──────────┐
                    │ BLEND           │     │  UPDATE  │ │  CREATE  │
                    │ embeddings:     │     │confidence│ │  NEW     │
                    │ V = (1-α)×V_old  │     │increase  │ │  NODE    │
                    │   + α×V_new      │     └──────────┘ └──────────┘
                    │                 │
                    │ α = 0.3         │
                    │ (30% new, 70%   │
                    │  old)           │
                    └─────────────────┘


EXAMPLE: MERGE (similarity = 0.97)
──────────────────────────────────────────────────────────────────────────

Existing: "deadline is Friday"
  embedding: [0.45, 0.23, -0.11, ...]
  confidence: 0.8

New: "deadline moved to Monday"
  embedding: [0.47, 0.21, -0.09, ...]
  confidence: 0.9

Action: MERGE
  Blended = (1-0.3) × [0.45, 0.23, -0.11, ...] + 0.3 × [0.47, 0.21, -0.09, ...]
          = [0.456, 0.226, -0.104, ...]  ← slightly shifted toward new

  Updated node:
    embedding: [0.456, 0.226, -0.104, ...]
    confidence: max(0.8, 0.9) = 0.9
    metadata: {previous_deadline: "Friday", current_deadline: "Monday"}


EXAMPLE: UPDATE (similarity = 0.82)
──────────────────────────────────────────────────────────────────────────

Existing: "project deadline"
  confidence: 0.7

New: "deadline is important"
  confidence: 0.85

Action: UPDATE
  confidence = max(0.7, 0.85) = 0.85
  embedding: unchanged (too different to blend)
  New edge created between nodes


EXAMPLE: CREATE NEW (similarity = 0.65)
──────────────────────────────────────────────────────────────────────────

Existing nodes: None similar enough

New: "buy coffee"
  embedding: [0.11, 0.88, 0.33, ...]

Action: CREATE NEW
  Add to database
  Create edges to similar existing nodes
  confidence = 0.85 (from extraction)
```

---

## Diagram 7: Graph Structure Visual

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     GRAPH STRUCTURE (Undirected)                      │
└─────────────────────────────────────────────────────────────────────────┘

                    [deadline_is_Friday]
                    role: FACT
                    activation: 0.87
                         │
              ┌──────────┼──────────┐
              │ 0.82     │ 0.91     │ 0.73
              │          │          │
      [finish_project] │    [stress_level]
      role: DECISION    │    role: FACT
      activation: 0.52  │    activation: 0.55
              │         │
              │ 0.65    │ 0.72
              │         │
      [project_report] │
      role: FACT       │
      activation: 0.48 │
                      │
                      │ 0.68
                      │
                [team_meeting]
                role: DECISION
                activation: 0.41


KEY INSIGHTS:
──────────────────────────────────────────────────────────────────────────

1. UNDIRECTED: Edges work both ways
   "deadline" activates "stress"
   "stress" also activates "deadline"

2. EDGE WEIGHTS: Stored on BOTH nodes
   deadline.neighbors = {finish_project: 0.82, stress: 0.91, ...}
   finish_project.neighbors = {deadline: 0.82, project_report: 0.65, ...}

3. MULTI-PARENT: Nodes can have multiple incoming paths
   "deadline" receives from:
   - finish_project (0.52)
   - stress (if it propagates back)
   Both contribute to final activation


BIDIRECTIONAL FLOW EXAMPLE:
──────────────────────────────────────────────────────────────────────────

Query: "project" (activates "finish_project")

Path 1 (forward):
  finish_project (0.52) → deadline (0.52 × 0.82 × 0.7 = 0.30)

Path 2 (backward, if deadline was also a Layer 1 match):
  deadline (0.87) → finish_project (0.87 × 0.82 × 0.7 = 0.50)

Result: Both directions contribute, reinforcing each other
```

---

## Diagram 8: What Happens in A Loop

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     LOOP HANDLING (Natural Decay)                     │
└─────────────────────────────────────────────────────────────────────────┘

   A bidirectional edge creates a potential loop:

      [computer_broke] ←────────→ [constipation]
              │                            │
              │ 0.85                       │ 0.85
              └────────────────────────────┘

   What happens when "computer_broke" gets activated (0.87)?

   ITERATION 1:
   ────────────
   computer_broke (0.87)
     → constipation: 0.87 × 0.85 × 0.7 = 0.52 ✓

   constipation (0.52)
     → computer_broke: 0.52 × 0.85 × 0.7 = 0.31 ✓
     But: 0.31 < 0.87 (existing), so no change

   ITERATION 2 (if we continued):
   ─────────────────────────────────
   computer_broke (still ~0.87)
     → constipation: 0.87 × 0.85 × 0.7 = 0.52 (same)

   constipation (0.52)
     → computer_broke: 0.52 × 0.85 × 0.7 = 0.31 (same)

   RESULT: System stabilizes naturally, no runaway!


DECAY PREVENTS RUNAWAY:
──────────────────────────────────────────────────────────────────────────

Without decay (hypothetical):
  computer_broke → constipation → computer_broke → constipation → ...
  0.87 → 0.74 → 0.63 → 0.53 → ... (eventually dies but takes forever)

With decay:
  0.87 → 0.52 → 0.31 → 0.18 → 0.11 → 0.07 → ... (dies quickly)


MULTIPLE HOPS WITH DECAY:
──────────────────────────────────────────────────────────────────────────

  [A] (0.87)
   │ decay=0.7
   ▼
  [B] (0.61)
   │ decay=0.7²=0.49
   ▼
  [C] (0.43)
   │ decay=0.7³=0.34
   ▼
  [A] (0.30)  ← came back to A, but much weaker!

  Even if there's a cycle, each traversal weakens the signal
```

---

## Diagram 9: SUM vs MAX for Multiple Parents

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MULTIPLE PARENT HANDLING                           │
└─────────────────────────────────────────────────────────────────────────┘

   A node can receive activation from multiple parents:

             [parent_1] (0.82)
                  │ 0.75
                  ├───┐
                  │   │
             [parent_2] (0.71)
                  │ 0.68
                  └───┐
                      ▼
                  [child]


CURRENT IMPLEMENTATION: SUM
──────────────────────────────────────────────────────────────────────────

child.activation = 0  (start)

From parent_1:
  delta = 0.82 × 0.75 × 1.0 × 0.7 = 0.43
  child.activation = 0 + 0.43 = 0.43

From parent_2:
  delta = 0.71 × 0.68 × 1.0 × 0.7 = 0.34
  child.activation = 0.43 + 0.34 = 0.77

Result: child.activation = 0.77 (reinforced by multiple parents)


ALTERNATIVE: MAX
──────────────────────────────────────────────────────────────────────────

child.activation = 0  (start)

From parent_1:
  delta = 0.43
  child.activation = max(0, 0.43) = 0.43

From parent_2:
  delta = 0.34
  child.activation = max(0.43, 0.34) = 0.43  (no change!)

Result: child.activation = 0.43 (only strongest path)


TRADEOFFS:
──────────────────────────────────────────────────────────────────────────

SUM (current):
  ✓ Reinforces nodes that receive multiple weak signals
  ✓ More inclusive retrieval
  ✗ Can overcount if many parents
  ✗ May activate irrelevant nodes

MAX (alternative):
  ✓ Preserves only strongest path
  ✓ Prevents overcounting
  ✗ Loses reinforcement effect
  ✗ May miss relevant connections

Current implementation uses SUM for inclusivity,
but can be toggled based on use case.
```

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     PARAMETER QUICK REFERENCE                          │
└─────────────────────────────────────────────────────────────────────────┘

┌────────────────────┬─────────┬─────────────────────────────────────────┐
│ PARAMETER          │ DEFAULT │ WHAT IT DOES                           │
├────────────────────┼─────────┼─────────────────────────────────────────┤
│ min_delta          │   0.3   │ Gate 1: Min signal to activate child  │
│ activation_thresh  │   0.5   │ Gate 2: Min activation to propagate    │
│ decay_per_hop      │   0.7   │ Retention per hop (0.7 = 70%)         │
│ propagation_depth │   2     │ Max hops to explore                     │
│ min_similarity     │   0.55  │ Min similarity for Layer 1 matches     │
│ blend_alpha        │   0.3   │ Embedding blend factor (merge)         │
└────────────────────┴─────────┴─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                     ROLE BOOSTS (KEY VALUES)                           │
└─────────────────────────────────────────────────────────────────────────┘

• GOAL → DECISION:      1.5×  (goals drive actions)
• CONSTRAINT → DECISION: -0.8× (constraints inhibit)
• FACT → FACT:          1.3×  (facts reinforce)
• DEFAULT:              1.0×  (no boost)

┌─────────────────────────────────────────────────────────────────────────┐
│                     COMMON TUNING SCENARIOS                            │
└─────────────────────────────────────────────────────────────────────────┘

TOO MUCH NOISE?
  min_delta: 0.3 → 0.4
  min_similarity: 0.55 → 0.65
  decay_per_hop: 0.7 → 0.6

NOT ENOUGH RESULTS?
  min_delta: 0.3 → 0.2
  activation_threshold: 0.5 → 0.4
  decay_per_hop: 0.7 → 0.8
  propagation_depth: 2 → 3

ONLY DIRECT MATCHES?
  activation_threshold: 0.5 → 0.3  (Gate 2 too strict)
```
