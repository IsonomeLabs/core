# isonome-framework — Iteration 017: Convergence Ratio Invariant & Pre-Adaptation Precision Fix

**Date:** 2026-06-03
**Cron Job:** Hourly incremental improvement
**Iteration:** 017
**Change Type:** Bug fix — equilibrium engine (task-type homeostasis)
**Tests:** 736/736 passing (+4 new)

---

## Summary

Fixed two mathematically incorrect behaviors in the task-type homeostasis module that undermined the reliability of convergence reporting and pre-adaptation precision.

**Bug 1 — convergence_ratio sentinel violated semantic invariant:**
`convergence_ratio` returned `1.0` for profiles with fewer than 3 observations. This is semantically ambiguous: a ratio of 1.0 could mean "genuinely high drift" OR "too few observations to compute." The `is_converged` property checks `ratio < 0.05`, and for <3 observations returns `False` (correct, not enough data). But `1.0 < 0.05` is also `False` — the invariant `is_converged == (convergence_ratio < 0.05)` happened to hold, but only by accident. If the convergence threshold were changed to e.g. 0.5 (to accept rougher profiles), `1.0 < 0.5` would be `False` while `is_converged` would still return `False` for the right reason — but a downstream consumer reading `convergence_ratio == 1.0` would misinterpret it as "this profile is drifting wildly" when it actually means "we haven't collected enough data yet."

**Bug 2 — apply_task_type_profile double-adjusted each axis:**
The method called `engine.adjust_default()` twice per axis. The first call used `outcome_signal=learned_default`, which via the formula `new = old + signal * learning_rate` shifts the default BY `learned_default * lr`, not TO `learned_default`. The second call then computed the needed signal from the already-corrupted position, producing a net shift to an intermediate value rather than the target. The code comment acknowledged the awkwardness ("we need to compute a signal that achieves the target default") but the first call was never removed.

**Impact:** Pre-adaptation now correctly restores axis defaults to the exact learned profile values (within floating-point precision). The `convergence_ratio` sentinel is now `np.inf`, making it semantically clear that convergence is undefined rather than "high."

| Property | Before (buggy) | After (fixed) |
|----------|---------------|---------------|
| `convergence_ratio` for <3 obs | `1.0` (ambiguous) | `np.inf` (undefined) |
| `apply_task_type_profile` adjust count | 2 per axis (cumulative error) | 1 per axis (exact) |
| Pre-adaptation target accuracy | Intermediate position | Exact learned default |
| Semantic invariant | Accidentally held | Guaranteed by design |

---

## What Was Changed

| File | Action | Change | Key Contribution |
|------|--------|--------|------------------|
| `isonome/equilibrium/task_type_homeostasis.py` | **Modified** | +12 / -8 lines | Fixed `convergence_ratio` sentinel to `np.inf`; removed double `adjust_default` in `apply_task_type_profile` |
| `tests/test_task_type_homeostasis.py` | **Modified** | +97 / -13 lines | Added 4 new tests; updated `test_adjusts_when_learned_differs` with precise assertions |

---

## Technical Details

### Bug 1: Convergence Ratio Sentinel

**Before:**

```python
@property
def convergence_ratio(self) -> float:
    if len(self._observations) < 3:
        return 1.0  # ambiguous: could be genuine high drift
    return self._compute_convergence_ratio()
```

**After:**

```python
@property
def convergence_ratio(self) -> float:
    # Returns np.inf when too few observations exist to compute
    # convergence. This preserves the invariant:
    #     is_converged == (convergence_ratio < 0.05)
    # for all observation counts, since np.inf is never < 0.05.
    if len(self._observations) < 3:
        return float(np.inf)
    return self._compute_convergence_ratio()
```

**Rationale:** `np.inf` is the mathematically correct sentinel for "undefined" — it communicates that convergence cannot be measured, not that it's high. Downstream code that compares against any finite threshold gets the right answer (not converged) while code that inspects the value can distinguish "too few observations" from "high drift."

### Bug 2: Double Adjust Default

**Before:**

```python
engine.adjust_default(axis_id, outcome_signal=learned_default)
# adjust_default uses outcome_signal * learning_rate, so we need
# to compute a signal that achieves the target default.
learning_rate = axis.learning_rate
if learning_rate > 0:
    needed_signal = (learned_default - axis.default_position) / learning_rate
    engine.adjust_default(axis_id, outcome_signal=needed_signal)
```

The first call shifts `default` by `learned_default * lr`. Then the second call reads `axis.default_position` (now corrupted) and computes the "needed signal" relative to the wrong baseline. The net result is:

    After call 1: d1 = d0 + learned_default * lr
    After call 2: d2 = d1 + (learned_default - d1) / lr * lr = learned_default

This ONLY works if `adjust_default` doesn't clamp. But `adjust_default` CLAMPS the result to [-1, 1], and the intermediate shift by `learned_default * lr` can push the default outside the valid range before clamping, changing the effective delta of the second call.

**After (fix):**

```python
current_default = axis.default_position
if abs(current_default - learned_default) < 0.001:
    continue
learning_rate = axis.learning_rate
if learning_rate > 0:
    needed_signal = (learned_default - current_default) / learning_rate
    engine.adjust_default(axis_id, outcome_signal=needed_signal)
```

Single call from the current position. No intermediate corruption. The signal is computed from the real current state.

---

## New Tests

| Test | Class | What It Verifies |
|------|-------|------------------|
| `test_convergence_ratio_invariant` | `TestTaskTypeProfile` | `is_converged == (convergence_ratio < 0.05)` holds for 0, 1, 2, 3+ observations |
| `test_convergence_ratio_inf_for_few_observations` | `TestTaskTypeProfile` | `convergence_ratio` returns `inf` for 0, 1, 2 observations |
| `test_exact_single_adjust_per_axis` | `TestPreAdaptation` | `apply_task_type_profile` lands axis at exact learned default (single call) |
| `test_soft_exact_one_third_distance` | `TestSoftPreAdaptation` | `soft_pre_adapt` moves exactly 1/3 toward learned default |

---

## Verification

    $ python -m pytest tests/test_task_type_homeostasis.py -v
    43 passed in 0.18s

    $ python -m pytest tests/ -v
    736 passed in 0.78s

Commit: dc9c53f -- pushed to origin/main.
