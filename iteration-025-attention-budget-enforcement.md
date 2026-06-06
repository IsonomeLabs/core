# Iteration 025: Attention Budget Enforcement

## Problem

The `AttentionEquilibriumSystem.add_chunk()` method blindly incremented
`tokens_used` without checking whether the context window budget had
room. There was no hard enforcement of the token capacity — a runaway
agent could accumulate context far beyond its window, with GC only
running retroactively when explicitly called.

This meant:
1. **Silent overflow**: `tokens_used` could exceed `token_capacity`
2. **No admission control**: every `add_chunk()` succeeded regardless
3. **No rejection feedback**: callers had no way to know a chunk was dropped
4. **No policy choice**: different deployment scenarios need different
   overflow strategies (reject vs. compress vs. auto-GC)

## Solution

Added `BudgetEnforcementPolicy` enum with three strategies:

| Policy | Behavior on overflow |
|--------|---------------------|
| `REJECT` | Silently drop new chunks that don't fit. No auto-GC. |
| `AUTO_GC` | Trigger GC to free space; reject if GC can't free enough. |
| `AUTO_COMPRESS` | Like AUTO_GC, but also compress the incoming chunk if still doesn't fit. |

### Key design decisions

1. **Oversized chunks always rejected**: A single chunk larger than
   total capacity is never admitted, regardless of policy.

2. **Enforcement threshold**: GC triggers at a configurable utilization
   fraction (default 0.85 = 85% full), not just at 100%. This gives
   the system headroom before hitting the hard wall.

3. **`add_chunk()` returns `Optional[AttentionChunk]`**: Returns `None`
   when a chunk is rejected. This is a **backward-compatible** change —
   existing code that doesn't check the return value still works, and
   new code can handle rejections gracefully.

4. **`enforce_budget()` public method**: Allows explicit enforcement
   checks separate from `add_chunk()`, useful for periodic maintenance.

5. **Enforcement statistics**: All enforcement events are tracked
   (auto-GC triggers, rejections, auto-compressions, oversized
   rejections, post-GC rejections) and exposed via `enforcement_stats`
   property and the `stats` dict.

### Changes

- `isonome/cognition/attention.py`:
  - `BudgetEnforcementPolicy` enum (REJECT, AUTO_GC, AUTO_COMPRESS)
  - `enforcement_policy` and `enforcement_threshold` __init__ params
  - `add_chunk()`: budget-aware admission with enforcement logic
  - `enforce_budget()`: public method for explicit budget checks
  - `enforcement_policy`, `enforcement_threshold`, `enforcement_stats` properties
  - Enforcement stats in `stats` dict
  - 6 enforcement counters tracked

- `isonome/cognition/__init__.py`: Export `BudgetEnforcementPolicy`

- `tests/test_attention_budget_enforcement.py`: 37 tests covering:
  - Enum values and members
  - `enforce_budget()` under/over capacity
  - `enforce_budget()` with REJECT policy
  - `add_chunk()` budget-aware admission for all 3 policies
  - Oversized chunk rejection
  - Post-GC rejection when GC can't free enough
  - Auto-compress when GC frees some but not enough space
  - Enforcement statistics tracking
  - Enforcement threshold configuration and clamping
  - Backward compatibility (default system still works)
  - Integration with calibration-aware GC

## Test Results

All 1285 tests pass (1248 existing + 37 new).

## Next Iteration Candidates

1. **Attention compression with semantic summarization**: Current
   compression is purely numerical (token_count * ratio). Real
   compression would summarize content while preserving key information.

2. **Priority queue for rejected chunks**: When a chunk is rejected,
   buffer it for retry after the next GC cycle.

3. **Budget-aware chunk splitting**: Break large chunks into smaller
   pieces that fit within remaining capacity.

4. **Adaptive enforcement threshold**: Adjust the enforcement threshold
   based on GC effectiveness — if GC consistently frees >50% capacity,
   raise the threshold to delay enforcement.
