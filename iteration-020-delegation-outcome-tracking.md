# isonome-framework — Iteration 020: Delegation Outcome Tracking

**Date:** 2026-06-04
**Cron Job:** Hourly incremental improvement
**Iteration:** 020
**Change Type:** feat (new system + integration tests + bug fix)
**Tests:** 1003/1003 passing (+46 new)

---

## Summary

Closed the metacognitive feedback loop in the delegation system by introducing **Delegation Outcome Tracking** with **Dynamic Threshold Adaptation**. When a delegated action's outcome is observed, it feeds back to the confidence calibrator as a (predicted_confidence, actual_success) pair, enabling the system to learn from its delegation decisions and improve future calibration. The ECE delegation threshold adapts automatically: when delegation accuracy is high (>0.8), the threshold tightens (fewer delegations); when accuracy is low (<0.5), it loosens (more delegations). This creates a self-correcting loop: poor calibration → delegation → outcome observation → calibrator update → improved calibration.

Also fixed a pre-existing bug where stress feedback tests incorrectly set axis positions that the engine constructor would reset to defaults.

## What Was Built

| File | Status | Description |
|------|--------|-------------|
| `isonome/praxis/delegation.py` | Modified | `DelegationOutcome` dataclass, `record_outcome()`, `_adapt_threshold()`, dynamic adaptation |
| `isonome/praxis/__init__.py` | Modified | Export `DelegationOutcome` |
| `tests/test_delegation_outcomes.py` | Created | 39 tests across 7 test classes |
| `tests/test_delegation.py` | Modified | Added `DelegationOutcome` import |
| `tests/test_cross_pillar_integration.py` | Modified | 7 cross-pillar integration tests for feedback loop |
| `tests/test_base_pillar.py` | Modified | Fix stress feedback tests (use `apply_feedback` to create drift) |

### Bug Fix

| Bug | Severity | Fix |
|-----|----------|-----|
| Stress feedback tests set axis `position=0.8` but `EquilibriumEngine.__init__` resets all positions to `default_position`, making drift=0 and stress=0 | High | Use `engine.apply_feedback()` to create real drift after construction |

## Architecture

```
                    Delegation Outcome Feedback Loop
                    ═════════════════════════════════

 ┌──────────────┐    check()     ┌──────────────┐    delegate    ┌──────────────┐
 │   Action     │ ──────────────→│  Delegation   │ ─────────────→│  Subagent /  │
 │  Orchestrator │               │     Gate      │               │  External    │
 └──────────────┘                └──────┬───────┘               └──────┬───────┘
                                        │                              │
                                        │ record_outcome()            │ outcome
                                        │                              │ observed
                                        │◄─────────────────────────────┘
                                        │
                                        ▼
                                ┌──────────────┐
                                │  Confidence   │
                                │  Calibrator   │
                                │  record()     │
                                └──────┬───────┘
                                       │
                                       │ ECE shifts
                                       ▼
                                ┌──────────────┐
                                │  Threshold    │
                                │  Adaptation   │
                                │  α = 0.02     │
                                └──────────────┘
```

### DelegationOutcome

A frozen dataclass recording the result of a delegated action:

| Field | Type | Description |
|-------|------|-------------|
| `action_id` | UUID | Links to the DelegationRecord |
| `action_description` | str | Human-readable action name |
| `action_risk` | int | Risk level at delegation time (0-4) |
| `predicted_confidence` | float | System's confidence when it delegated |
| `actual_success` | bool | Whether the delegated action succeeded |
| `delegated_mode` | DelegationMode | Mode that triggered delegation |
| `ece_at_delegation` | float | ECE when delegation was decided |
| `feedback_to_calibrator` | bool | Whether to feed back (default True) |

### Dynamic Threshold Adaptation

```
  delegation_accuracy = recent_successful / recent_total (rolling window)

  if accuracy > 0.8:  θ_delegate *= (1 − α)   # tighten → fewer delegations
  if accuracy < 0.5:  θ_delegate *= (1 + α)   # loosen  → more delegations
  else:               no change               # sweet spot

  Bounds: θ_delegate ∈ [0.05, 0.5]
  α = 0.02 (conservative, prevents oscillation)
  Window: 50 recent outcomes (configurable)
```

**Rationale:** When most delegated actions succeed, the system is being too cautious — it should trust itself more and delegate less. When most fail, it's not being cautious enough — it should delegate more aggressively. The adaptation rate is intentionally conservative to prevent oscillation.

### Integration with Calibrator

`record_outcome()` calls `calibrator.record(predicted_confidence, actual_success)`, directly closing the metacognitive loop:

- **Delegated action succeeds despite low confidence** → calibrator receives (low_c, True) → accuracy at low confidence bins increases → ECE may decrease → delegation mode may improve
- **Delegated action fails** → calibrator receives (high_c, False) → accuracy at high confidence bins decreases → ECE increases → delegation is confirmed as necessary

## Test Coverage

### test_delegation_outcomes.py (39 tests)

| Class | Tests | What it covers |
|-------|-------|----------------|
| TestDelegationOutcome | 4 | Frozen dataclass, fields, defaults, slots |
| TestRecordOutcome | 8 | Counter increments, calibrator feedback, skip when disabled, no calibrator edge case |
| TestDelegationAccuracy | 6 | Rolling window computation, 0%/100%/mixed accuracy, window sliding |
| TestThresholdAdaptation | 9 | No adaptation with <3 outcomes, tighten/loosen/no-op, floor/ceiling bounds, conservative rate, reset |
| TestOutcomeSerialization | 8 | to_dict includes outcomes/stats/config, round-trip preservation, malformed data |
| TestOutcomeStats | 2 | Stats dict includes all outcome fields, accuracy matches property |
| TestOutcomeRepr | 1 | repr includes outcome count, accuracy, threshold |
| TestOutcomeWithOrchestrator | 1 | Orchestrator serialization includes outcome data |

### test_cross_pillar_integration.py (+7 tests)

| Test | What it covers |
|------|----------------|
| test_successful_outcome_feeds_calibrator | Real calibrator gets prediction pair |
| test_failed_outcome_feeds_calibrator | Real calibrator gets failure pair |
| test_outcome_feedback_changes_ece | ECE shifts after corrective feedback |
| test_high_accuracy_tightens_threshold | Threshold decreases |
| test_low_accuracy_loosens_threshold | Threshold increases |
| test_feedback_disabled_does_not_affect_calibrator | No update when feedback=False |
| test_full_delegation_loop_serialization | Round-trip with outcomes and adapted threshold |

## Cross-Pillar Impact

| Pillar | Impact |
|--------|--------|
| **Cognition** | Calibrator receives delegation outcomes as calibration signals, improving confidence estimates |
| **Praxis** | DelegationGate now self-improves via threshold adaptation; outcome tracking provides audit trail |
| **Mneme** | Future: delegated action outcomes can be tagged "delegated" for pattern matching |

## Design Decisions

1. **Conservative adaptation rate (α=0.02):** Prevents threshold oscillation. A single outcome changes the threshold by at most 2%, requiring sustained patterns to significantly shift delegation behavior.

2. **Threshold bounds [0.05, 0.5]:** Floor 0.05 = well-calibrated boundary (never drop below it). Ceiling 0.5 = half the ECE range (extreme delegation would be counterproductive — if ECE > 0.5, the system is so miscalibrated that delegating everything isn't enough).

3. **Rolling window (default 50):** Prevents ancient outcomes from anchoring the adaptation. Recent outcomes are more representative of current system state.

4. **feedback_to_calibrator flag:** Allows recording outcomes for audit without affecting the calibrator. Useful for external monitoring or when the outcome source is unreliable.

5. **Separate DelegationOutcome from DelegationRecord:** Records capture the *decision* (what was decided and why); Outcomes capture the *result* (what actually happened). This separation of concerns keeps both data models clean.

## Lines of Code

| Module | Before | After | Delta |
|--------|--------|-------|-------|
| isonome/praxis/delegation.py | 408 | 564 | +156 |
| tests/test_delegation_outcomes.py | 0 | 374 | +374 |
| tests/test_cross_pillar_integration.py | 551 | 733 | +182 |
| **Total** | | | **+712** |
