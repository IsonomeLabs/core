# isonome-framework — Iteration 015: Adaptive Damping Controller

**Date:** 2026-06-02
**Cron Job:** Hourly incremental improvement
**Iteration:** 015
**Change Type:** feat + fix (new system + 3 bug fixes)
**Tests:** 704/704 passing (+75 new)

---

## Summary

Introduced the `AdaptiveDampingController` — a per-axis dynamic damping system that resolves the fundamental tension between responsiveness (low damping) and stability (high damping). When an axis oscillates, damping automatically increases to stiffen the feedback loop; when stability is sustained, damping gradually decays back to the axis's base value, restoring responsiveness. The controller integrates transparently with `EquilibriumEngine` via an opt-in flag, serializes for cross-session persistence, and uncovered three bugs during test-driven development that were fixed in this iteration.

## What Was Built

| File | Status | Description |
|------|--------|-------------|
| `isonome/equilibrium/__init__.py` | Modified | `AdaptiveDampingController` class + engine integration |
| `tests/test_adaptive_damping.py` | Created | 75 tests across 9 test classes |

### Bug Fixes (discovered during test development)

| Bug | Severity | Fix |
|-----|----------|-----|
| Validation order: `damping_min >= damping_max` checked before range checks, giving wrong error for out-of-range values | Medium | Reordered: range checks first, then min<max |
| Decay anchor gap: effective damping could drift below `base_damping` on sustained stability | Medium | Added `else` branch clamping to `base_damping` when decay reaches floor |
| Unregistered axis KeyError: `on_feedback` for unregistered axis with stable history crashed on return | High | Auto-register axis on first `on_feedback` contact |

## Architecture

```
                     ┌─────────────────────────────────┐
                     │      EquilibriumEngine           │
                     │                                 │
                     │  apply_feedback()               │
                     │    ├─ read adaptive_damping      │
                     │    │  .effective_damping()       │
                     │    ├─ axis.adjust(delta)         │
                     │    │  with adaptive damping      │
                     │    └─ notify adaptive_damping    │
                     │       .on_feedback()             │
                     │                                 │
                     │  apply_feedback_batch()          │
                     │    ├─ Phase 1: compute updates   │
                     │    │  (use adaptive damping)     │
                     │    ├─ Phase 2: apply updates     │
                     │    └─ Phase 3: notify controller │
                     │                                 │
                     │  to_dict() / from_dict()         │
                     │    └─ includes                   │
                     │       adaptive_damping_state     │
                     └──────────┬──────────────────────┘
                                │
                ┌───────────────▼───────────────┐
                │  AdaptiveDampingController     │
                │                               │
                │  per-axis state:              │
                │  ├─ _effective_damping{}      │
                │  ├─ _stability_counters{}     │
                │  └─ _oscillation_severity{}   │
                │                               │
                │  on_feedback(axis, history):  │
                │  ├─ severity = stddev / θ     │
                │  ├─ if severity > 1.0:        │
                │  │    d_eff += boost×severity  │
                │  │    reset stability counter  │
                │  └─ else:                      │
                │       counter++                │
                │       if counter ≥ window:     │
                │         d_eff -= decay_rate    │
                │         anchor to base_damping │
                │                               │
                │  Bounds: [d_min, d_max]        │
                │  Auto-register on first touch  │
                └───────────────────────────────┘
```

## Core Mechanisms

### 1. Oscillation Severity

$$
\text{severity} = \frac{\sigma}{\theta_{\text{osc}}}
$$

Where $\sigma$ is the population standard deviation of the last $N$ position values and $\theta_{\text{osc}}$ is the oscillation threshold. Severity > 1.0 indicates oscillation.

### 2. Boost on Oscillation

$$
d_{\text{eff}} \leftarrow \min\!\left(d_{\max},\; d_{\text{eff}} + r_{\text{boost}} \cdot \min(\text{severity},\; 2.0)\right)
$$

The boost is proportional to severity (capped at 2.0×) and clamped by `damping_max`. Oscillation resets the stability counter to zero.

### 3. Decay on Sustained Stability

$$
d_{\text{eff}} \leftarrow \max\!\left(d_{\min},\; d_{\text{eff}} - r_{\text{decay}}\right)
$$

After `stability_window` consecutive stable ticks. An anchor mechanism prevents damping from decaying below `base_damping`:

- If $d_{\text{eff}} > d_{\text{base}}$: apply extra decay pull toward base
- If $d_{\text{eff}} \leq d_{\text{base}}$: clamp to $d_{\text{base}}$ (floor)

This ensures effective damping converges to the axis's static value during calm periods, never below it.

### 4. Auto-Registration

Previously, `on_feedback` for an unregistered axis would crash with `KeyError` when trying to return the effective damping (the stable branch doesn't write to `_effective_damping` unless the stability window is met). Fixed by auto-registering any axis on first `on_feedback` contact, setting its effective damping to `base_damping` and initializing counters to zero.

## Tension Modulation

The adaptive damping controller is **orthogonal** to tension positions — it modulates the *damping coefficient* that controls how much each feedback signal moves an axis position, not the position itself. This creates a second-order effect:

| Axis State | Effective Damping | Behavior |
|-----------|-------------------|----------|
| Oscillating | Increased (up to d_max) | Feedback moves position less → stabilizes faster |
| Stable | Decreased toward base | Feedback moves position more → responsive to new signals |

The controller does NOT compose additively with tension positions (unlike calibration modulation in iter-008/009). It replaces the static damping value used inside `TensionAxis.adjust()` before the position update is computed.

## Test Coverage

| Test Class | Tests | Coverage |
|-----------|-------|----------|
| TestAdaptiveDampingConstruction | 15 | Parameter validation, defaults, custom values, edge cases |
| TestAxisRegistration | 8 | Register, unregister, overwrite, fallback values |
| TestOscillationBoost | 6 | Boost on oscillation, severity scaling, clamping, cap at 2.0 |
| TestStabilityDecay | 6 | Counter increment, window threshold, damping_min floor, base anchor |
| TestSeverityComputation | 6 | Short/constant/varying history, threshold ratio, boundary |
| TestReset | 5 | Clear all state, re-registration requirement |
| TestAdaptiveDampingSerialization | 5 | Empty/populated round-trip, key preservation, defaults, JSON |
| TestEngineAdaptiveDampingIntegration | 14 | Engine with/without, custom controller, single/batch feedback, serialization, reset |
| TestFullOscillationStabilizeCycle | 3 | End-to-end oscillate→boost→stabilize→decay, bounds, axis independence |
| TestEdgeCases | 9 | Unregistered axis, damping at min/max, window=1, interleaved, snapshot isolation |
| **Total** | **75** | |

## Design Decisions

1. **Opt-in, not default**: `enable_adaptive_damping=False` by default. Existing code behavior is unchanged. The controller must be explicitly enabled or passed in.

2. **Per-axis independence**: Each axis has its own effective damping, stability counter, and severity. Oscillation on one axis doesn't affect another. This avoids cross-contamination.

3. **Auto-registration on first contact**: Rather than requiring all axes to be pre-registered, `on_feedback` auto-registers unknown axes. This prevents crashes and makes the controller resilient to engine changes (adding/removing axes).

4. **Base damping as floor**: During decay, effective damping is anchored to the axis's static `base_damping` — it never goes below. This prevents the adaptive system from making axes *more* responsive than their designed static value. The adaptive controller adds rigidity when needed but never removes the baseline.

5. **Severity capped at 2.0**: Extremely high severity (e.g., position wildly swinging) produces a bounded boost. This prevents a single pathological tick from maxing out damping.

6. **Stability window before decay**: Decay only activates after `stability_window` consecutive stable ticks. This prevents premature relaxation when oscillation is intermittent (alternating oscillation and brief stability).

7. **Validation order fix**: Range checks (`[0,1]`) now execute before the min<max comparison, ensuring out-of-range parameters get the correct error message instead of a misleading "min must be < max" error.

8. **Serialized as optional**: `adaptive_damping_state` is `None` when the controller is not enabled. `from_dict` handles both cases gracefully, restoring the controller only when state is present.

## Why This Creates Impact

**Short-term**: The engine's feedback loops are now self-stabilizing. Any axis that starts oscillating (from aggressive feedback, changing environment, or pillar misconfiguration) will automatically stiffen until it calms down, then relax back. This eliminates a common failure mode where oscillating tension axes cascade into erratic agent behavior.

**Long-term**: Adaptive damping is a prerequisite for multi-agent equilibrium (research direction #4), where shared tension axes could cause cross-agent oscillation. The per-axis independence and serialization support make it ready for fleet-level deployment. The controller also provides a measurable signal (`oscillation_severity`, `total_adaptations`) that higher-level systems can use to detect environmental instability.

## Architecture: Full Framework State

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          IsonomeAgent                                    │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│  │  CognitionPillar │  │   PraxisPillar   │  │   MnemePillar    │          │
│  │  (νοῦς)          │  │  (πρᾶξις)        │  │  (μνήμη)         │          │
│  │                  │  │                  │  │                  │          │
│  │  AttentionSys ◄─┼──┼─► ActionOrch     │  │  HierMneme       │          │
│  │  ReasoningEng   │  │  DAG executor    │  │  3-tier memory   │          │
│  │  ConfCalibrator │  │  Safety gates    │  │  Calib-aware     │          │
│  │  Calib-amp(8)   │  │  Calib-gate(6)   │  │  Calib-thresh(9) │          │
│  │  Calib-attn(8)  │  │  Calib-verify(9a)│  │  Calib-reh(10)   │          │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
│           │                    │                    │                    │
│    ┌──────▼────────────────────▼────────────────────▼──────┐            │
│    │              EquilibriumEngine                         │            │
│    │                                                        │            │
│    │  8 tension axes ──────────────────────────────┐       │            │
│    │  ├─ Cognition: explore_exploit, shallow_deep, │       │            │
│    │  │  divergent_convergent                       │       │            │
│    │  ├─ Praxis: autonomy_safety, sequential_par,  │       │            │
│    │  │  verify_execute                             │       │            │
│    │  └─ Mneme: consolidate_prune, specific_general│       │            │
│    │                                                │       │            │
│    │  Oscillation detection ────────────────────────┤       │            │
│    │  Outcome-driven homeostasis (iter-012) ────────┤       │            │
│    │  Task-type adaptive homeostasis (iter-013) ────┤       │            │
│    │  Pillar equilibrium pull (iter-014) ───────────┤       │            │
│    │                                                │       │            │
│    │  ┌─────────────────────────────────────────┐   │       │            │
│    │  │ AdaptiveDampingController (iter-015) ✨  │◄──┘       │            │
│    │  │                                         │           │            │
│    │  │  Per-axis: d_eff, stability, severity   │           │            │
│    │  │  Boost on oscillation, decay on calm     │           │            │
│    │  │  Base-damping floor, [d_min, d_max]      │           │            │
│    │  │  Auto-register, serialization, reset      │           │            │
│    │  └─────────────────────────────────────────┘           │            │
│    └────────────────────────────────────────────────────────┘            │
│                                                                          │
│  Cross-pillar pipelines:                                                 │
│  νοῦς→πρᾶξις  πρᾶξις→μνήμη  πρᾶξις→νοῦς  νοῦς→μνήμη  μνήμη→νοῦς     │
│                                                                          │
│  Metacognitive loops:                                                    │
│  Calibrator→all  SafetyGate→Praxis  VerifyDepth→Praxis                   │
│  AttentionRetention→Cognition  ConsolidationThreshold→Mneme              │
│  RehearsalBoost→Mneme  PatternSupport→Mneme  ImportFloor→Mneme           │
│  HomeostaticLearning→Engine  TaskTypeAdapt→Engine  PillarPull→Engine     │
│  AdaptiveDamping→Engine ✨                                              │
└──────────────────────────────────────────────────────────────────────────┘
```

## Next Iteration Candidates

1. **Calibration-gated delegation**: When ECE exceeds threshold, delegate to subagent — uses calibrator to detect when the agent's own reasoning is unreliable and should be offloaded
2. **Attention weight rebalancing on overconfidence**: Shift from task-relevance (β) to surprisal (α) when overconfident, making the system seek novelty when its confidence is unreliable
3. **Calibration-based rehearsal scheduling**: Frequency-based rehearsal (not just boost magnitude) — how often to rehearse, not just how much to boost
4. **Adaptive damping threshold learning**: Make `oscillation_threshold` adaptive per-axis based on each axis's typical variance, rather than a single global value
5. **Multi-agent calibration pooling**: Shared calibrator parameters across fleet agents for faster convergence

## Files Changed

```
isonome/
  equilibrium/
    __init__.py          # +AdaptiveDampingController class, engine integration, 3 bug fixes
tests/
  test_adaptive_damping.py  # NEW — 75 tests
```

## Commit Stack

```
fix(equilibrium): reorder validation checks — range before min<max
fix(equilibrium): decay anchor — clamp effective damping to base_damping floor
fix(equilibrium): auto-register axis on first on_feedback contact
feat(equilibrium): AdaptiveDampingController — per-axis dynamic damping
feat(equilibrium): integrate AdaptiveDampingController into EquilibriumEngine
test: 75 tests for AdaptiveDampingController and engine integration
docs: iteration-015 — adaptive damping controller
```
