# Iteration 014 — Pillar-Equilibrium Pull Mechanism

**Date:** 2025-06-02
**Pillar:** Equilibrium (cross-cutting: Cognition, Praxis, Mneme)
**Type:** Feature (core architecture)
**Impact:** High — both immediate (pillars now homeostatically self-regulate) and long-term (every pillar becomes equilibrium-aware by default)

---

## Summary

Implemented the **Pillar-Equilibrium Pull Mechanism** — a closed-loop homeostatic feedback system where each pillar automatically reads its equilibrium state and emits corrective feedback when stressed. This transforms the equilibrium system from a passive recipient of push-based feedback into an active governor that pillars *pull* from on every tick.

Before this change, pillars only pushed feedback *to* the engine (via `apply_feedback`). They had no way to *read* equilibrium state, detect stress, or self-correct. The pull mechanism closes the loop: pillars now auto-sync their equilibrium view each tick, detect when their own axes are far from homeostasis, and emit low-confidence corrective feedback that gently pulls them back toward balance.

---

## Problem

The isonome framework had an **open-loop** tension management architecture:

1. **Push-only:** Pillars could only *push* `Feedback` into the `EquilibriumEngine`. They never read the engine's state.
2. **No self-regulation:** A pillar that pushed its own axes far from homeostasis had no mechanism to detect or correct the drift.
3. **Blind to cross-pillar state:** Each pillar had no visibility into what other pillars were doing to the shared equilibrium space.
4. **Manual sync only:** `update_tension_profile()` required explicit calls; nothing auto-synced.

This meant that a cognition pillar that kept pushing `explore_exploit` toward 1.0 would never feel the pull back toward 0.15 (its default). The system could drift unchecked.

---

## Solution

### 1. `PillarEquilibriumView` (equilibrium/__init__.py, +224 lines)

A read-only projection of the engine's state, scoped to a single pillar. The view decomposes the engine's 8 axes into:

- **`own_axes`** — axes owned by this pillar (e.g., Cognition sees `explore_exploit`, `shallow_deep`, `divergent_convergent`)
- **`cross_axes`** — axes owned by other pillars (e.g., Cognition sees `autonomy_safety`, `consolidate_prune`, etc.)
- **`all_positions`** / **`all_defaults`** — current and default positions for all 8 axes
- **`drift`** — per-axis distance from default (signed: positive = right of default, negative = left)
- **`stress_level`** — RMS drift across all axes: `sqrt(1/N * Σ(drift²))`
- **`is_stressed`** — `stress_level > 0.3`
- **`is_highly_stressed`** — `stress_level > 0.6`
- **`oscillating`** — tuple of axis IDs where recent position history shows oscillation (direction reversal rate > threshold)
- **`is_axis_oscillating(id)`** — check a specific axis
- **`summary()`** — dict for logging/debugging
- **Convenience:** `get(id, default)`, `own_axis_ids()`, `cross_axis_ids()`, `get_drift(id)`

All dict properties return **copies** (defensive — mutations don't leak back to the engine).

### 2. `EquilibriumEngine.view_for(pillar)` (equilibrium/__init__.py, +3 lines)

Single method on the engine that creates a `PillarEquilibriumView` for any `Pillar` enum member. O(1) — just reads current axis state, no history walk.

### 3. `BasePillar` engine binding (base.py, +173 lines)

#### New attributes:
- `_engine: EquilibriumEngine | None` — bound engine reference
- `_equilibrium_view: PillarEquilibriumView | None` — cached view
- `_stress_feedback_enabled: bool` — toggle (default `True`)

#### New methods:
- **`bind_engine(engine)`** — Binds the engine, creates the initial view. Raises if already bound to a different engine. Idempotent if same engine.
- **`unbind_engine()`** — Clears engine and view. Allows re-binding to a different engine.
- **`equilibrium_view`** property — Returns cached view (or None if unbound).
- **`_on_equilibrium_sync(view)`** — Hook for subclasses. Called after each auto-sync. Default is no-op.
- **`_emit_stress_feedback(view)`** — Core homeostatic pull logic:
  1. Find the own-axis with maximum absolute drift
  2. Compute a corrective signal: `drift * -0.2 * min(stress, 1.0)`
  3. Emit a `Feedback` with `confidence = stress * 0.3` (low confidence — gentle nudge, not a command)
  4. Reason: `"stress-reactive: own axis '{id}' drifted {drift:.2f} from default {default:.2f}"`
  5. Only fires when `stress > 0.3` and `stress_feedback_enabled`

#### Modified `process_queued()`:
After draining and processing signals, if bound:
1. **Auto-sync:** `self._equilibrium_view = self._engine.view_for(self.pillar)`
2. **Hook:** `self._on_equilibrium_sync(self._equilibrium_view)`
3. **Stress feedback:** `self._emit_stress_feedback(self._equilibrium_view)`

### 4. Pillar-specific `_on_equilibrium_sync()` overrides

#### CognitionPillar (cognition/pillar.py, +73 lines)
- Calls `reasoning.set_tension_profile(view.all_positions)` to update the reasoning engine's tension profile
- Logs own-axis and cross-axis positions at DEBUG level
- Cross-axis awareness: reads `autonomy_safety`, `consolidate_prune` to inform reasoning depth/explore decisions

#### PraxisPillar (praxis/pillar.py, +25 lines)
- Calls `orchestrator.set_tension_profile(view.all_positions)` to update execution strategy
- Logs axis positions
- Cross-axis awareness: reads `explore_exploit` to calibrate action selection risk

#### MnemePillar (mneme/pillar.py, +37 lines)
- Calls `mneme.set_tension_profile(view.all_positions)` to update memory strategy
- Logs axis positions
- Cross-axis awareness: reads `explore_exploit`, `shallow_deep`, `autonomy_safety` to calibrate consolidation vs. pruning

---

## Architecture Diagram

```
                    ┌─────────────────────────┐
                    │   EquilibriumEngine      │
                    │  ┌─────┐ ┌─────┐ ┌────┐ │
                    │  │ E/E │ │ S/D │ │V/E │ │  ← Cognition axes
                    │  ├─────┤ ├─────┤ ├────┤ │
                    │  │ A/S │ │ S/P │ │V/E │ │  ← Praxis axes
                    │  ├─────┤ ├─────┤      │ │
                    │  │ C/P │ │ S/G │      │ │  ← Mneme axes
                    │  └─────┘ └─────┘──────┘ │
                    └─────────┬───────────────┘
                              │ view_for(pillar)
                    ┌─────────┼───────────────┐
                    │         ▼               │
          ┌─────────┤  PillarEquilibriumView  ├─────────┐
          │         │  own_axes │ cross_axes  │         │
          │         │  stress   │ oscillating │         │
          │         └─────────┬───────────────┘         │
          │                   │                         │
     ┌────▼────┐        ┌────▼────┐              ┌────▼────┐
     │Cognition│        │ Praxis  │              │ Mneme   │
     │         │        │         │              │         │
     │ _on_    │        │ _on_    │              │ _on_    │
     │  equili-│        │  equili-│              │  equili-│
     │  brium_ │        │  brium_ │              │  brium_ │
     │  sync() │        │  sync() │              │  sync() │
     │         │        │         │              │         │
     │ _emit_  │        │ _emit_  │              │ _emit_  │
     │  stress_│        │  stress_│              │  stress_│
     │  feed-  │        │  feed-  │              │  feed-  │
     │  back() │───────▶│  back() │─────────────▶│  back() │
     │         │ push   │         │ push         │         │
     └─────────┘        └─────────┘              └─────────┘
          │                   │                         │
          └───────────────────┴─────────────────────────┘
                       Feedback.apply_feedback()
                              │
                              ▼
                    ┌─────────────────────────┐
                    │   EquilibriumEngine      │  ← CLOSED LOOP
                    └─────────────────────────┘
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **View is a copy, not a reference** | Prevents mutations from leaking back to the engine. Pillar reads are side-effect-free. |
| **Stress = RMS drift across all 8 axes** | Uses the global state, not just own-axes. A pillar is stressed when the *system* is stressed. |
| **Stress feedback targets max-drift own-axis** | Most efficient homeostatic strategy — fix the worst deviation first. |
| **Low confidence (0.09–0.18) for stress feedback** | These are gentle nudges, not commands. They can be overridden by high-confidence deliberate feedback. |
| **Signal = drift × -0.2 × min(stress, 1.0)** | Proportional to both drift and stress. The `min(stress, 1.0)` cap prevents runaway correction. |
| **Only fires above stress 0.3** | Below 0.3, the system is within normal operating range — no correction needed. |
| **`_on_equilibrium_sync()` is a hook, not mandatory** | Subclasses override it; `BasePillar` default is no-op. This is backward-compatible. |
| **`bind_engine()` raises on double-bind** | Prevents accidental re-binding to a different engine. Must `unbind_engine()` first. |
| **`update_tension_profile()` still works** | Push and pull coexist. The pull mechanism auto-syncs, but manual push is still available. |

---

## Test Coverage

**58 new tests** in `tests/test_pillar_equilibrium_pull.py` (782 lines), organized into 14 test classes:

| Class | Tests | What it covers |
|-------|-------|----------------|
| `TestPillarEquilibriumViewAxisDecomposition` | 8 | own vs. cross axis split for all 3 pillars |
| `TestPillarEquilibriumViewStress` | 4 | stress computation, thresholds, RMS formula |
| `TestPillarEquilibriumViewDrift` | 3 | per-axis drift, convenience method |
| `TestPillarEquilibriumViewOscillation` | 3 | oscillation detection |
| `TestPillarEquilibriumViewConvenience` | 7 | `get()`, `summary()`, `repr()`, copy safety, `pillar` property |
| `TestBasePillarBindEngine` | 6 | bind/unbind lifecycle, idempotent, error on double-bind |
| `TestBasePillarAutoSync` | 4 | auto-sync on tick, no sync without engine, stress feedback trigger |
| `TestStressReactiveFeedback` | 6 | target axis, confidence, signal direction, bounds, per-pillar axis mapping |
| `TestCognitionEquilibriumSync` | 2 | hook fires, cross-axes visible |
| `TestPraxisEquilibriumSync` | 2 | hook fires, cross-axes visible |
| `TestMnemeEquilibriumSync` | 2 | hook fires, cross-axes visible |
| `TestFullPullLoopIntegration` | 4 | end-to-end: bind→tick→stress→feedback→tick, 3-pillar bind, cross-pillar influence |
| `TestViewWithCustomAxes` | 3 | non-default axis configurations, empty axes |
| `TestBackwardCompatibility` | 4 | `update_tension_profile()` still works, push+pull coexist |

**Total suite: 629/629 passed** (571 existing + 58 new).

---

## Files Changed

| File | Lines | Delta | What changed |
|------|-------|-------|-------------|
| `isonome/equilibrium/__init__.py` | 670 | +248 | `PillarEquilibriumView` class (224 lines), `view_for()` method |
| `isonome/base.py` | 309 | +173 | `bind_engine()`, `unbind_engine()`, `equilibrium_view`, `_on_equilibrium_sync()`, `_emit_stress_feedback()`, auto-sync in `process_queued()`, `import math` |
| `isonome/cognition/pillar.py` | 661 | +73 | `_on_equilibrium_sync()` override with reasoning profile update + cross-axis logging |
| `isonome/praxis/pillar.py` | 393 | +25 | `_on_equilibrium_sync()` override with orchestrator profile update |
| `isonome/mneme/pillar.py` | 264 | +37 | `_on_equilibrium_sync()` override with mneme profile update + cross-axis logging |
| `tests/test_pillar_equilibrium_pull.py` | 782 | +782 | 58 tests across 14 classes |

**Total: +555 lines production code, +782 lines tests = 1,337 lines added.**

---

## Immediate Impact

1. **Self-regulation:** Pillars now automatically pull back toward homeostasis when stressed. No manual intervention needed.
2. **Cross-pillar awareness:** Each pillar can see what other pillars are doing to the shared equilibrium space.
3. **Oscillation detection:** The view flags axes that are oscillating, enabling future dampening strategies.

## Long-Term Impact

1. **Every future pillar is equilibrium-aware by default.** Just override `_on_equilibrium_sync()` — the binding, auto-sync, and stress feedback are all in `BasePillar`.
2. **The closed loop enables emergent homeostasis.** The system can now self-balance without external control.
3. **Foundation for meta-cognition.** The stress level and oscillation data can feed into higher-level decision-making (e.g., "I'm highly stressed → switch to conservative mode").
4. **Telemetry-ready.** `view.summary()` provides structured data for dashboards and logging.

---

## Backward Compatibility

- All 571 existing tests pass unchanged.
- `update_tension_profile()` continues to work (push mechanism).
- `bind_engine()` is opt-in — pillars work identically without it.
- The `_on_equilibrium_sync()` hook is a no-op by default in `BasePillar`.
