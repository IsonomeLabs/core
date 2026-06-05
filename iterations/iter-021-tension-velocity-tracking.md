# iter-021: Tension Velocity Tracking & Momentum-Aware Restoration

**Date**: 2026-06-04
**Status**: Complete
**Test Coverage**: 59 new tests, 1062 total (1003 baseline + 59 velocity)
**Backward Compatibility**: Full — velocity tracking is opt-in

## Problem

The EquilibriumEngine tracked position history for oscillation detection but had no awareness of the *rate* of position change (velocity). This created three gaps:

1. **Late oscillation detection**: Oscillation was declared only after position stddev exceeded a threshold — by then, the system was already oscillating. No early warning.
2. **Blind restoration**: The engine applied uniform restoring forces regardless of whether an axis was moving toward or away from its default. No momentum awareness.
3. **No velocity data for consumers**: PillarEquilibriumView provided positions and drift but no rate-of-change information that pillars could use for smarter behavior modulation.

## Solution

### New Class: `TensionVelocityTracker` (`isonome/equilibrium/velocity.py`)

A per-axis velocity tracker that computes:

- **Velocity**: `v[i] = position[i] - position[i-1]` (simple finite difference)
- **Reversal**: `sign(v[i]) != sign(v[i-1])` with both `|v[i]|` and `|v[i-1]|` ≥ `min_reversal_magnitude`
- **Reversal rate**: Recent reversals / possible reversal points in window
- **Momentum score**: `velocity × (default - position)`: positive = heading home, negative = drifting away
- **Oscillation prediction**: `is_oscillation_imminent()` — reversal rate > threshold (predictive, unlike stddev-based post-hoc detection)

Key design decisions:
- `__slots__` for consistency with existing codebase patterns
- Auto-registration of unknown axes on `on_position_update()`
- Configurable `window_size` and `min_reversal_magnitude`
- Full serialization via `to_dict()`/`from_dict()`

### Engine Integration (`isonome/equilibrium/__init__.py`)

**EquilibriumEngine changes:**
- New constructor params: `velocity_tracker`, `enable_velocity_tracking`
- `apply_feedback()`: calls `tracker.on_position_update()` after storing new axis
- `apply_feedback_batch()`: Phase 3 velocity update before Phase 4 adaptive damping
- `reset()`: resets and re-registers all axes with tracker
- `to_dict()`/`from_dict()`: full serialization round-trip
- New property: `velocity_tracker`

**PillarEquilibriumView changes:**
- New constructor param: `velocity_tracker`
- New `__slots__`: `_velocities`, `_momentum_scores`, `_oscillation_imminent`
- New properties: `velocities`, `momentum_scores`, `oscillation_imminent`
- New convenience methods: `get_velocity()`, `get_momentum_score()`, `is_axis_drifting()`
- `summary()` includes velocity data when available
- `__repr__` shows `vel_warn=N` when oscillation imminent axes exist

### Reversal Detection Fix

Fixed a bug in the initial implementation where reversals were compared against `_prev_velocity` (velocity from two updates ago) instead of the current stored `_velocity` (velocity from the previous update). This caused reversal detection to miss the first sign change after steady movement.

## Architecture

```
Feedback → EquilibriumEngine.apply_feedback()
  ├── axis.adjust(delta)           # position update
  ├── tracker.on_position_update() # velocity tracking
  └── adaptive_damping.on_feedback() # damping adjustment

Pillar.tick()
  ├── engine.view_for(pillar)     # read equilibrium state
  │   └── PillarEquilibriumView
  │       ├── positions, drift, stress  (existing)
  │       ├── velocities, momentum      (NEW)
  │       └── oscillation_imminent      (NEW — predictive)
  └── pillar.behavior()           # modulated by view
```

## Usage

```python
# Enable velocity tracking on engine creation
engine = EquilibriumEngine(enable_velocity_tracking=True)

# Or provide a custom tracker
tracker = TensionVelocityTracker(window_size=20, min_reversal_magnitude=0.01)
engine = EquilibriumEngine(velocity_tracker=tracker)

# Velocity data is automatically available in pillar views
view = engine.view_for(Pillar.COGNITION)
if view.is_axis_drifting("exploration_exploitation"):
    # Axis is moving away from default — increase restoring force
    ...
if view.oscillation_imminent:
    # Preemptive damping — oscillation predicted before stddev exceeds threshold
    ...

# Direct tracker access
velocity = engine.velocity_tracker.get_velocity("shallow_deep")
momentum = engine.velocity_tracker.get_momentum_score("shallow_deep")
is_approaching = engine.velocity_tracker.is_approaching_default("shallow_deep")
```

## Test Coverage

- `TestVelocityTrackerConstruction` (4 tests): validation, defaults, repr
- `TestVelocityTrackerRegistration` (3 tests): register, unregister, auto-register
- `TestVelocityComputation` (7 tests): zero first velocity, positive/negative velocity, multiple updates, all_velocities, unknown axis
- `TestMomentumScore` (7 tests): approaching default, drifting from default, at-default zero, all scores, unknown axis
- `TestReversalDetection` (5 tests): no reversal, single reversal, multiple reversals, magnitude threshold, total reversals
- `TestReversalRateAndPrediction` (6 tests): insufficient data, no reversals, computed rate, oscillation imminent/not, custom threshold, unknown axis
- `TestWindowRollover` (1 test): small window boundary
- `TestVelocityTrackerReset` (1 test): full reset verification
- `TestVelocityTrackerSerialization` (2 tests): round-trip, empty round-trip
- `TestEngineIntegration` (10 tests): with/without tracker, explicit tracker, axis registration, apply_feedback, batch, reset, serialization, backward compatibility
- `TestPillarEquilibriumViewVelocity` (7 tests): view with/without tracker, convenience methods, summary, repr
- `TestAdaptiveDampingVelocityCoexistence` (2 tests): both enabled, momentum-aware pattern
- `TestEdgeCases` (3 tests): zero signal, confidence levels, update count

## Cross-Pillar Impact

- **Cognition**: Can use `momentum_scores` to modulate reasoning depth — coast when axes are heading home, dig deeper when drifting
- **Praxis**: Can use `is_axis_drifting()` to increase autonomy_safety restoring force
- **Metacognition**: Can use `oscillation_imminent` for preemptive self-regulation before oscillation is declared

## Next Steps

- **Velocity-aware adaptive damping**: Wire `is_oscillation_imminent()` into `AdaptiveDampingController.on_feedback()` for preemptive damping boost (currently only position stddev triggers damping increase)
- **Momentum-modulated restoring force**: Add a configurable restoring force multiplier based on momentum sign (stronger when drifting, weaker when approaching)
- **Cross-agent calibration pooling**: Research topic still open from research-directions.md
