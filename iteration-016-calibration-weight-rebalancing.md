# isonome-framework — Iteration 016: Calibration-Driven Weight Rebalancing

**Date:** 2026-06-03
**Cron Job:** Hourly incremental improvement
**Iteration:** 016
**Change Type:** feat (new mechanism + integration + tests)
**Tests:** 732/732 passing (+26 new)

---

## Summary

Added `_compute_calibration_weight_rebalance()` to `AttentionEquilibriumSystem` — a metacognitive mechanism that shifts the α↔β attention weight balance in response to calibration state. When the ConfidenceCalibrator detects overconfidence (ECE > 0.15), the system boosts α (surprisal/novelty weight) and reduces β (MI/coherence weight), pushing the attention system to seek diverse, unexpected inputs. When underconfidence is detected, β is boosted and α is reduced, reinforcing trust in coherent, well-established information. The shift is zero-sum (α_delta + β_delta = 0), capped at 0.12, and overconfidence receives a 1.2× amplification bonus reflecting the higher risk of confident errors. The rebalance integrates additively into `_modulate_weights()` alongside existing tension-based modulations, and GC reports now include weight rebalance deltas with `calWΔ` notation in summaries.

## What Was Built

| File | Status | Description |
|------|--------|-------------|
| `isonome/cognition/attention.py` | Modified | `_compute_calibration_weight_rebalance()` + integration in `_modulate_weights()` and `collect_garbage()` |
| `tests/test_calibration_attention.py` | Modified | 26 new tests across 3 test classes |

## Mechanism Design

### Weight Rebalance Formula

```
η = 0.50                     # base learning rate
max_shift = 0.12             # hard cap on weight delta

if ECE < 0.15:               # well-calibrated: no shift
    Δ = 0

elif is_overconfident:       # boost surprisal (α↑ β↓)
    Δ = min(η × ECE × (1 + |bias|) × 1.2, max_shift)
    α_delta = +Δ
    β_delta = -Δ

elif is_underconfident:      # boost MI/coherence (α↓ β↑)
    Δ = min(η × ECE × (1 + |bias|), max_shift)
    α_delta = -Δ
    β_delta = +Δ

else:                        # moderate miscalibration, no direction
    Δ = 0
```

### Key Properties

1. **Zero-sum invariant**: `α_delta + β_delta = 0` always — total weight budget preserved before normalization
2. **Overconfidence amplification**: 1.2× multiplier because overconfident systems pose higher risk (they act on wrong beliefs with certainty)
3. **Hard cap at 0.12**: prevents runaway weight distortion even under extreme miscalibration
4. **Bias amplification**: `(1 + |bias|)` factor means higher confidence bias → stronger rebalance signal
5. **Additive composition**: calibration rebalance is added to tension-based weight shifts before final normalization, so both mechanisms compose cleanly

### Integration Points

```
 ┌───────────────────────────────────────────┐
 │  AttentionEquilibriumSystem               │
 │                                           │
 │  _modulate_weights(profile)               │
 │  ├─ base weights (α=0.35, β=0.35, ...)   │
 │  ├─ tension-based shifts (±0.10)          │
 │  │  (shallow/deep, explore/exploit, ...)  │
 │  ├─ **calibration rebalance (±Δ)**        │  ← NEW
 │  │  (_compute_calibration_weight_rebalance)│
 │  └─ normalize to sum=1.0                  │
 │                                           │
 │  collect_garbage()                        │
 │  ├─ calibration retention modifier        │
 │  ├─ calibration decay modifier            │
 │  └─ **GC report: calWΔ α=+0.12/β=-0.12** │  ← NEW
 └───────────────────────────────────────────┘
```

### Example Scenarios

| Calibration State | ECE | Bias | α_delta | β_delta | Rationale |
|---|---|---|---|---|---|
| Inactive | — | — | 0.0 | 0.0 | No calibration data |
| Well-calibrated | 0.10 | 0.05 | 0.0 | 0.0 | ECE below threshold |
| Overconfident | 0.16 | 0.05 | +0.1008 | -0.1008 | Moderate shift toward novelty |
| Overconfident | 0.20 | 0.10 | +0.12 | -0.12 | Capped shift (raw=0.132) |
| Underconfident | 0.20 | 0.10 | -0.11 | +0.11 | Shift toward coherence (no 1.2×) |
| Extreme overconfident | 0.50 | 0.50 | +0.12 | -0.12 | Hard cap prevents distortion |

## Test Coverage

### TestCalibrationWeightRebalance (16 tests)
- Inactive/zero ECE/below threshold: 4 boundary tests
- Overconfident: 6 tests (direction, exact calculation, cap, zero bias, negative bias, extreme ECE)
- Underconfident: 4 tests (direction, exact calculation, comparison with overconfident, extreme ECE)
- Moderate miscalibration: 1 test (no shift without direction flag)
- Zero-sum invariant: 1 parametrized test across 5 cases

### TestGCReportWeightRebalance (7 tests)
- Report fields default to 0.0 without calibration
- Overconfident/underconfident report correct signed deltas
- Low ECE reports 0.0
- Summary includes `calWΔ` when active, excludes when inactive

### TestModulateWeightsWithCalibration (5 tests)
- Overconfident increases α in final weights
- Underconfident increases β in final weights
- Weights still sum to 1.0 after rebalance
- Calibration composes with tension-based modulation
- No rebalance without over/underconfident flag

## Design Decisions

1. **Why additive, not multiplicative?** Multiplicative rebalance would interact non-linearly with tension shifts — an explore tension × 1.2× overconfident bonus could overshoot. Additive is predictable and composes cleanly.

2. **Why 1.2× for overconfidence only?** Overconfident systems are dangerous because they act with certainty on wrong beliefs. Underconfident systems are conservative (hesitant but not wrong). The asymmetry reflects risk: we want to *urgently* diversify when overconfident, but gently reinforce coherence when underconfident.

3. **Why 0.12 cap?** At default weights (α=0.35, β=0.35), a 0.12 shift produces α=0.47 or α=0.23 — a meaningful but not dominant rebalance. Beyond 0.12 the attention system would essentially ignore one weight entirely.

4. **Why zero-sum?** Weight rebalance is a *redistribution* of the attention budget, not an expansion. This preserves the equilibrium engine's normalization invariant and prevents the rebalance from inflating total weight.
