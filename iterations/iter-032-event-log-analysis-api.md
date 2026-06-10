# Iteration 032: Event Log Analysis API

## Summary

Add structured analysis methods to `TensionEventLog` that pillars can use for
adaptive behavior. Previously the engine recorded rich events but provided no
way to query actionable analytics beyond simple counts and timelines.

## Motivation

The event log (added in iter-023) stores detailed records of feedback,
oscillation, cooldown, and convergence events. However, the only analysis
methods available were basic aggregations: `stress_timeline()`,
`feedback_density()`, `count_by_type()`, etc. Pillars needed higher-level
analytics to make informed decisions:

- **Which pillar is stressing the system the most?** (pillar_stress_scores)
- **Is an axis moving erratically?** (axis_volatility)
- **Is there a feedback burst happening?** (detect_feedback_bursts)
- **Who dominates feedback on a contested axis?** (dominant_feedback_source)
- **Is the system converging or diverging?** (detect_convergence_from_events)
- **Are two pillars fighting over an axis?** (detect_cross_pillar_conflicts)

These analytics enable pillars to make adaptive decisions: back off during
bursts, modulate behavior when in conflict, or change strategy when
diverging.

## Changes

### `isonome/equilibrium/event_log.py`

Six new methods on `TensionEventLog`:

1. **`pillar_stress_scores()`** → `dict[Pillar, float]`
   - Cumulative `|delta| * confidence` per pillar across all feedback events
   - Measures how much force each pillar is exerting on the system

2. **`axis_volatility()`** → `dict[TensionID, float]`
   - Standard deviation of `position_after` values per axis
   - High volatility = erratic axis; low = stable
   - Engine-wide events (empty axis_id) excluded

3. **`detect_feedback_bursts(window=5, threshold=3)`** → `list[dict]`
   - Detects rapid consecutive feedback on a single axis within a tick window
   - Returns burst descriptors with axis_id, tick range, and event count
   - Only considers FEEDBACK_APPLIED events

4. **`dominant_feedback_source()`** → `dict[TensionID, dict]`
   - For each axis, identifies the pillar with highest cumulative weight
   - Returns pillar, total_weight, and event_count per axis

5. **`detect_convergence_from_events()`** → `dict`
   - Linear regression on |delta| trend over feedback events
   - Returns direction (converging/diverging/stable/unknown), confidence,
     and trend_slope
   - OSCILLATION_DETECTED events bias toward divergence

6. **`detect_cross_pillar_conflicts()`** → `list[dict]`
   - Detects same-axis feedback from different pillars in opposing directions
   - Returns conflict descriptors with pillars, opposing_deltas, and
     conflict_intensity (0.0–1.0)
   - Same-pillar opposing feedback is NOT a cross-pillar conflict

### `isonome/equilibrium/__init__.py`

- Added `event_log` property to `PillarEquilibriumView` so pillars can
  access analysis methods through their scoped view

### `tests/test_event_log_analysis.py`

38 new tests covering all six analysis methods:
- `TestPillarStressScores` (5 tests)
- `TestAxisVolatility` (5 tests)
- `TestDetectFeedbackBursts` (6 tests)
- `TestDominantFeedbackSource` (4 tests)
- `TestDetectConvergenceFromEvents` (6 tests)
- `TestDetectCrossPillarConflicts` (5 tests)
- `TestEngineIntegration` (4 tests)

## Test Results

```
1872 passed (including 38 new analysis tests)
```

## Design Decisions

1. **Analysis methods are on the log, not the engine**: This keeps the
   EquilibriumEngine focused on feedback processing and the TensionEventLog
   focused on event analytics. The engine already has its own convergence
   detector; the log's `detect_convergence_from_events()` provides an
   event-based perspective that complements it.

2. **Only FEEDBACK_APPLIED events contribute to stress/burst/dominant
   analysis**: Other event types (oscillation, cooldown, reset) are signals
   about the system's response, not the pillar's intent.

3. **Burst detection reports first burst per axis**: To avoid noise from
   overlapping windows, only the first detected burst per axis is reported.

4. **Conflict intensity is normalized to [0, 1]**: `total_force / 2.0` capped
   at 1.0 gives an intuitive measure where 1.0 means maximum disagreement.

5. **PillarEquilibriumView gets an event_log property**: Rather than adding
   each analysis method to the view separately, providing access to the log
   itself keeps the view thin while enabling full analytics.
