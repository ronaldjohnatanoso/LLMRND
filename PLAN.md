# Implementation Plan: Tiered Propagation Limits + Neuroplasticity

## Overview
Add adaptive propagation limits and Hebbian learning to make the cognitive graph learn from experience.

---

## Phase 1: Tiered Quality-Based Propagation Limits

### Files
- **New**: `cog_memory/propagation_config.py`
- **Modified**: `cog_memory/query_interface.py`

### Implementation
1. Create `PropagationConfig` dataclass:
   - Tier thresholds (excellent/good/moderate/weak/poor → neighbor limits)
   - Role modifiers (Goals: 1.5x, Observations: 0.8x)
   - Hop decay (deeper hops = fewer neighbors)
   - Hard cap multiplier (prevent explosion)

2. Add `get_limit(node, hop)` method:
   - Calculate quality score (70% peak + 30% average of neighbor weights)
   - Find appropriate tier
   - Apply role modifier
   - Apply hop decay
   - Apply hard cap

3. Update propagation loop:
   - Calculate limit for each node before iterating neighbors
   - Only propagate to top K strongest neighbors

### Example
```python
# Node with strong connections (max_weight=0.95, avg=0.82)
# Quality: 0.95*0.7 + 0.82*0.3 = 0.921 → "excellent" tier → 15 neighbors
# Role modifier: 15 * 1.0 = 15
# Hop decay (hop=1): 15 * 0.5 = 7

# Node with weak connections (max_weight=0.45, avg=0.32)
# Quality: 0.45*0.7 + 0.32*0.3 = 0.411 → "weak" tier → 5 neighbors
```

---

## Phase 2: Connection Usage Tracking

### Files
- **Modified**: `cog_memory/node.py`

### Implementation
1. Add attributes to Node:
   - `connection_usage: dict[str, int]` - usage counter per neighbor
   - `connection_last_used: dict[str, float]` - timestamp per neighbor

2. Add methods:
   - `use_connection(neighbor_id)` - track usage during propagation
   - `get_connection_stats(neighbor_id)` - get usage statistics

3. Update `to_dict()` to include usage stats for persistence

### Example
```python
node.use_connection("neighbor_123")
# connection_usage = {"neighbor_123": 1}
# connection_last_used = {"neighbor_123": 1707220800.0}
```

---

## Phase 3: Hebbian Learning (Neuroplasticity)

### Files
- **New**: `cog_memory/plasticity_config.py`
- **New**: `cog_memory/neuroplasticity.py`
- **Modified**: `cog_memory/query_interface.py`

### Implementation

1. Create `PlasticityConfig`:
   - `learning_rate: float = 0.02` - how much to strengthen per use
   - `min_weight: float = 0.05` - minimum connection weight
   - `max_weight: float = 1.0` - maximum connection weight
   - `decay_enabled: bool = False` - whether to decay unused connections
   - `consolidation_enabled: bool = True` - periodic consolidation

2. Create `NeuroplasticityManager`:
   - Track query count
   - `learn_from_query(propagation_history)` - apply Hebbian learning
   - `_consolidate_memory()` - strengthen/prune connections

3. Update `query_simulation()`:
   - Track propagation events in list
   - Call `neuroplasticity.learn_from_query()` after propagation
   - Return learning stats in response

### Learning Formula
```python
weight_change = activation_delta × learning_rate
new_weight = clamp(old_weight + weight_change, min_weight, max_weight)
```

### Example
```python
# Connection: Cai Lun → mulberry (weight=0.84)
# Query: "how is paper made"
# Activation delta: 0.664
# Learning rate: 0.02
# New weight: 0.84 + (0.664 × 0.02) = 0.853
```

---

## Phase 4: Memory Consolidation

### Files
- **Modified**: `cog_memory/neuroplasticity.py`

### Implementation

Add `_consolidate_memory()` method:
1. Run every N queries (configurable, default 100)
2. For each connection:
   - If usage_count >= 10: strengthen (weight × 1.1)
   - If usage_count < 2 AND weight < 0.3: prune (delete)
3. Reset usage counters after consolidation

### Example
```python
# Before consolidation (after 100 queries)
connection_usage = {"mulberry": 87, "wood_pulp": 76, "asbestos": 3}
connection_weights = {"mulberry": 0.94, "wood_pulp": 0.88, "asbestos": 0.52}

# After consolidation
connection_weights = {"mulberry": 0.99, "wood_pulp": 0.93}  # asbestos pruned
```

---

## Phase 5: Backend API Integration

### Files
- **Modified**: `backend/main.py`

### Implementation

1. Add Pydantic models:
   - `PropagationConfigRequest`
   - `PlasticityConfigRequest`
   - Update `QueryRequest`

2. Update `/query/simulation` endpoint:
   - Accept propagation and plasticity configs
   - Pass through to query_simulation

3. Add learning stats to response

### Example Request
```json
{
  "query_text": "how is paper made",
  "propagation_config": {
    "hop_decay_enabled": true,
    "hard_cap_multiplier": 2.0
  },
  "plasticity_config": {
    "learning_rate": 0.02,
    "consolidation_enabled": true
  },
  "enable_plasticity": true
}
```

---

## Phase 6: Persistence Layer

### Files
- **Modified**: `cog_memory/query_interface.py`

### Implementation

1. Add `save_learned_weights(filepath)`:
   - Save all node neighbors (connection weights)
   - Save usage stats
   - Save query count

2. Add `load_learned_weights(filepath)`:
   - Load connection weights
   - Load usage stats
   - If file doesn't exist: use default weights (normal behavior)

3. Auto-save after consolidation

### File Format
```json
{
  "metadata": {
    "timestamp": 1707220800.0,
    "query_count": 150,
    "last_consolidation": 1707210000.0
  },
  "nodes": {
    "node_id_123": {
      "neighbors": {"neighbor_456": 0.95, "neighbor_789": 0.82},
      "connection_usage": {"neighbor_456": 87, "neighbor_789": 23},
      "connection_last_used": {"neighbor_456": 1707220800.0}
    }
  }
}
```

### Important: Weight Loading Behavior
**If no learned weights file exists:**
- System uses default weights from ingestion (normal behavior)
- No error, no difference from current system
- Learning starts from first query

**If learned weights file exists:**
- Load saved weights and usage stats
- Continue learning from where left off
- Query count and consolidation state preserved

---

## Phase 7: UI Enhancements

### Files
- **Modified**: `src/app/page.tsx`

### Implementation

1. Show connection weights in node details:
   - Display weight for each neighbor
   - Show usage count
   - Show weight change indicator (↑0.05, ↓0.02)

2. Add learning stats panel:
   - Total queries
   - Connections strengthened (this query)
   - Connections pruned (total)
   - Next consolidation in: X queries

3. Visual indicators:
   - Color code strong vs weak connections
   - Show learned vs initial weights

---

## Phase 8: Testing

### Files
- **New**: `tests/test_neuroplasticity.py`

### Test Cases

1. `test_tiered_propagation_limits()`:
   - Verify nodes with strong connections get higher limits
   - Verify role modifiers apply
   - Verify hop decay works

2. `test_hebbian_learning()`:
   - Run query, verify weights increase
   - Verify min/max constraints respected

3. `test_consolidation()`:
   - Run 100 queries
   - Verify consolidation runs
   - Verify weak connections pruned
   - Verify strong connections strengthened

4. `test_persistence()`:
   - Save weights
   - Create new instance
   - Load weights
   - Verify weights preserved

5. `test_no_weights_file()`:
   - Start system without weights file
   - Verify normal behavior
   - Verify no errors

---

## Configuration

### Default PropagationConfig
```python
tiers = {
    "excellent": (0.90, 15),
    "good": (0.75, 10),
    "moderate": (0.60, 7),
    "weak": (0.45, 5),
    "poor": (0.30, 3),
    "noise": (0.0, 1),
}
role_modifiers = {
    Role.GOAL: 1.5,
    Role.DECISION: 1.2,
    Role.OBSERVATION: 0.8,
    others: 1.0,
}
hop_decay_factor = 0.5
hard_cap_multiplier = 2.0
```

### Default PlasticityConfig
```python
learning_rate = 0.02
min_weight = 0.05
max_weight = 1.0
decay_enabled = False
consolidation_enabled = True
consolidation_interval = 100
```

---

## Migration Path

### For Existing Systems
1. Deploy with new code (plasticity disabled by default)
2. System works exactly as before (backward compatible)
3. Enable plasticity when ready
4. Learned weights saved to `data/learned_weights.json`
5. Weights auto-load on restart

---

## Success Criteria

- [x] Tiered limits prevent exponential explosion
- [x] Connections strengthen with use (Hebbian learning)
- [x] Weak connections pruned during consolidation
- [x] Weights persist across restarts
- [x] No errors if weights file missing
- [x] UI shows learning progress
- [x] All tests pass

---

## Estimated Time
- Phase 1: 30 min (PropagationConfig)
- Phase 2: 15 min (Usage tracking)
- Phase 3: 45 min (Hebbian learning)
- Phase 4: 30 min (Consolidation)
- Phase 5: 20 min (API integration)
- Phase 6: 20 min (Persistence)
- Phase 7: 30 min (UI updates)
- Phase 8: 30 min (Testing)

**Total: ~4 hours**
