# isonome-framework — Iteration 018: Calibration-Based Rehearsal Scheduling

**Date:** 2026-06-04
**Cron Job:** Hourly incremental improvement
**Iteration:** 018
**Change Type:** feat (new system) + fix (syntax, formula)
**Tests:** 821/821 passing

---

## Summary

Adds calibration-aware rehearsal *scheduling* (WHEN to rehearse) to complement the existing calibration-gated rehearsal *boosting* (HOW MUCH to boost, iter-010). The RehearsalScheduler computes optimal intervals using significance, effective half-life, calibration state, and tension. Two new HierarchicalMneme methods (`get_rehearsal_candidates`, `rehearse_due_candidates`) integrate scheduling with the existing memory system. Also fixes three bugs from the incomplete previous run: missing `def set_calibration_state` method signature, inverted significance formula, and inverted tension modifier sign.

## What Was Built

| File | Change |
|------|--------|
| `isonome/mneme/hierarchical.py` | +RehearsalScheduler class, +get_rehearsal_candidates, +rehearse_due_candidates, formula fixes |
| `isonome/mneme/__init__.py` | Export RehearsalScheduler |
| `tests/test_rehearsal_scheduling.py` | 27 tests across 6 classes |

## Architecture

```
                    RehearsalScheduler
                    ┌──────────────────┐
                    │ set_calibration  │◄── MnemePillar.set_calibration_state()
                    │ compute_interval │
                    │ compute_next_at  │
                    │ is_due           │
                    │ to_dict/from_dict│
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │ HierarchicalMneme│
                    │ get_rehearsal_   │──► sorted due entries (urgency)
                    │   candidates     │
                    │ rehearse_due_    │──► auto-rehearse all due entries
                    │   candidates     │
                    └──────────────────┘
```

## Core Mechanisms

### Interval Formula

```
I_base = effective_half_life × significance_factor
I_cal  = I_base × (1 + calibration_modifier)
I_tens = I_cal × (1 + tension_modifier)
I_final = clamp(I_tens, MIN_INTERVAL, MAX_INTERVAL)
```

Where:
- `effective_half_life = base_half_life × 1.5^rehearsal_count` (inherited from MemoryEntry)
- `significance_factor = 0.5 + significance` (range 0.5–1.5, higher sig → longer interval)
- `calibration_modifier`: well-calibrated=+0.10, overconfident=−0.25, underconfident=+0.20
- `tension_modifier = 0.15 × consolidate_prune_position`
- MIN_INTERVAL=300s (5 min), MAX_INTERVAL=86400s (24h)

### Tension Direction

- `consolidate_prune < 0` (Consolidate): tension_modifier negative → shorter intervals → rehearse more
- `consolidate_prune > 0` (Prune): tension_modifier positive → longer intervals → let decay

### Calibration Guard

≥10 predictions required (same as iter-010). Below guard, calibration_modifier=0.

## Tension Modulation

Multiplicative composition: `(1 + cal_modifier) × (1 + tension_modifier)`. Consistent with iter-008/009 additive-with-tension pattern for continuous modulation.

## Test Coverage

| Class | Tests | Coverage |
|-------|-------|----------|
| TestRehearsalSchedulerBasic | 8 | Interval computation, clamping, significance, expansion, due-check |
| TestRehearsalSchedulerCalibration | 6 | Over/under/well-calibrated modes, modifier values, prediction guard |
| TestRehearsalSchedulerTension | 3 | Consolidate/prune/combined effects |
| TestMnemeGetRehearsalCandidates | 6 | Empty mneme, stale/recent entries, tier filter, max limit, urgency sort |
| TestMnemeRehearseDueCandidates | 2 | Count, post-rehearsal not due |
| TestRehearsalSchedulerSerialization | 3 | Round-trip, mneme integration, missing data fallback |
| **Total** | **27** | |

## Design Decisions

1. **Significance factor = 0.5 + significance** (not reciprocal): High-significance entries are genuinely stable, so they get longer intervals. The reciprocal formula was inverted and contradicted the docstring.

2. **No separate rehearsal_expansion**: effective_half_life already includes 1.5^rehearsal_count from MemoryEntry. Adding another 1.3^count would double-count the spacing effect.

3. **Tension modifier sign = +0.15 × position**: The negative sign was inverted — consolidate (neg position) needs shorter intervals, which requires a negative modifier, so the multiplier must be positive.

4. **Multiplicative calibration×tension composition**: Matches the continuous-modulation pattern from iter-008/009. Discrete step-function gates (iter-010) are not used here since interval timing is a continuous knob.

## Bugs Fixed (from incomplete previous run)

1. **Missing method signature**: `def set_calibration_state(self, ...)` was absent — `__init__` body flowed directly into parameter list, causing SyntaxError
2. **Inverted significance formula**: `1/(0.5+sig)` made high-significance entries get shorter intervals (opposite of intent)
3. **Inverted tension modifier**: `-0.15 × position` made consolidate→longer and prune→shorter (opposite of intent)

## Next Iteration Candidates

- Calibration-gated delegation (delegate to subagent when ECE exceeds threshold)
- Attention weight rebalancing when overconfident (shift β→α)
- Cross-agent calibration pooling (shared calibrator parameter)

## Files Changed

```
isonome/mneme/hierarchical.py     (+250 lines: RehearsalScheduler, integration methods, fixes)
isonome/mneme/__init__.py         (+2 lines: RehearsalScheduler export)
tests/test_rehearsal_scheduling.py (+406 lines: 27 tests, 6 classes)
```

## Commit Stack

1. `fix(mneme): repair RehearsalScheduler syntax — add missing set_calibration_state method signature`
2. `fix(mneme): correct significance factor and tension modifier formulas in RehearsalScheduler`
3. `feat(mneme): add get_rehearsal_candidates and rehearse_due_candidates integration methods`
4. `test: add 27 tests for iter-018 calibration-based rehearsal scheduling`
5. `docs: add iteration-018 serialized MD`
