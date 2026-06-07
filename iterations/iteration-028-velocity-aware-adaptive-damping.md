# Iteration 028: Velocity-Aware Adaptive Damping

**Date**: 2026-06-07
**Status**: Complete
**Depends on**: iter-021 (tension-velocity-tracking)

## Summary

Wire `TensionVelocityTracker.is_oscillation_imminent()` into
`AdaptiveDampingController.on_feedback()` for preemptive damping
boosts — damping increases *before* position stddev confirms oscillation,
using velocity reversal rates as an early warning signal.

## Problem

The existing `AdaptiveDampingController` is purely reactive: it detects
oscillation only after position history stddev exceeds the threshold
(severity > 1.0). By the time the controller boosts damping, the axis
has already been oscillating for several ticks. This introduces
unnecessary overshoot and prolongs the oscillation before stabilization.

Meanwhile, iteration-021 introduced `TensionVelocityTracker` with
`is_oscillation_imminent()` — a velocity-based predictor that detects
high reversal rates *before* position stddev crosses the threshold.
The two features were independent and not integrated.

## Solution

### AdaptiveDampingController extensions

- **New constructor parameters**:
  - `velocity_tracker: TensionVelocityTracker | None = None` — bind a
    tracker for preemptive detection
  - `preemptive_boost_rate: float | None = None` — damping increase on
    imminent oscillation (defaults to `boost_rate / 2` for a gentler,
    preventive boost vs. the full reactive boost)
  - `preemptive_threshold: float = 0.4` — reversal rate threshold passed
    to `is_oscillation_imminent()`

- **on_feedback() priority order**:
  1. Position-based oscillation (severity > 1.0): full `boost_rate * severity`
  2. Velocity-based imminent oscillation: gentler `preemptive_boost_rate`
  3. Stability: decay toward base damping

- **New properties**:
  - `velocity_tracker` (getter + setter) — bind/unbind at runtime
  - `preemptive_boost_rate` (read-only)
  - `preemptive_oscillation_count` (read-only) — total preemptive boosts

- **Serialization**: `to_dict()` / `from_dict()` preserve
  `preemptive_oscillation_count` and `preemptive_boost_rate`

### EquilibriumEngine auto-wiring

When both `enable_velocity_tracking=True` and
`enable_adaptive_damping=True` are passed to `EquilibriumEngine`, the
engine automatically wires the velocity tracker to the damping
controller. This also works on `from_dict()` deserialization and
`reset()`.

## Mathematical model

```
On oscillation:      d_eff = min(d_max, d_eff + boost_rate * severity)
On imminent (vel):   d_eff = min(d_max, d_eff + preemptive_boost_rate)
On stability:        d_eff = max(d_min, d_eff - decay_rate)
```

Where `preemptive_boost_rate` defaults to `boost_rate / 2`.

## Tests

44 new tests in `tests/test_velocity_aware_damping.py`:

- `TestPreemptiveDampingBoost` (6 tests): core preemptive behavior
- `TestPreemptiveThresholdConfig` (4 tests): threshold parameterization
- `TestVelocityTrackerBinding` (4 tests): constructor / setter wiring
- `TestPreemptiveVsPositionBased` (3 tests): priority ordering
- `TestPositionVsVelocityInteraction` (4 tests): interaction scenarios
- `TestPreemptiveSerializationRoundTrip` (4 tests): serialization
- `TestPreemptiveDecay` (4 tests): decay after preemptive boost
- `TestPreemptiveWithMultipleAxes` (4 tests): multi-axis isolation
- `TestPreemptiveConfiguration` (4 tests): config edge cases
- `TestPreemptiveWithEngineIntegration` (4 tests): engine-level wiring
- `TestBackwardCompatibility` (3 tests): no regression without tracker

Updated `tests/test_adaptive_damping.py` to expect new serialization keys.

Full suite: **1535 passed**.

## Files changed

| File | Change |
|------|--------|
| `isonome/equilibrium/__init__.py` | `AdaptiveDampingController` + `EquilibriumEngine` wiring |
| `tests/test_velocity_aware_damping.py` | **New**: 44 tests |
| `tests/test_adaptive_damping.py` | Updated serialization key set |

## Next steps

- **iter-029**: Evidential decay integration with velocity-weighted trust
  scoring — use velocity trajectory to weight how quickly evidence
  for a pillar position decays under contradiction
- **iter-030**: Dashboard visualization for preemptive vs. reactive
  damping events over time
