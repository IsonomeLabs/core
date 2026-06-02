# isonome-framework — Iteration 009: Calibration-Aware Mneme Consolidation

**Date:** 2026-06-02
**Cron Job:** Hourly incremental improvement
**Iteration:** 009
**Change Type:** Calibration cross-pillar feedback — Mneme pillar metacognition
**Tests:** 437/437 passing (+26 new)

---

## Summary

The Mneme pillar now reads calibration metrics from the reasoning engine's `ConfidenceCalibrator` and modulates its consolidation and pruning behavior accordingly. This closes the **final cross-pillar calibration feedback loop**, completing the metacognitive architecture across all three pillars of the isonome framework.

- **Overconfident agents** (high ECE, positive bias) consolidate more cautiously — raising thresholds so only clearly significant memories promote to higher tiers
- **Overconfident agents prune less aggressively** — when the system cannot trust its relevance judgments, it keeps borderline memories instead of discarding them
- **Well-calibrated agents** see minimal modulation — consolidation proceeds efficiently with established tension-based thresholds
- **Calibration data flows automatically** from Cognition → Mneme every tick via the `CognitionPillar.update_tension_profile()` metacognitive push

## What Was Built

| File | Action | Change | Description |
|------|--------|--------|-------------|
| `isonome/mneme/hierarchical.py` | **Modified** | +128 / −3 | `set_calibration_state()` method, calibration-aware `_modulate_thresholds()`, overconfident pruning discount, `ConsolidationReport` calibration fields |
| `isonome/mneme/pillar.py` | **Modified** | +24 / −0 | `update_calibration()` method forwarding to `HierarchicalMneme` with ≥10 prediction guard |
| `isonome/cognition/pillar.py` | **Modified** | +19 / −0 | Optional `mneme_pillar` reference; pushes calibration to Mneme each tick via `update_tension_profile()` |
| `tests/test_calibration_mneme.py` | **Created** | +604 | 26 tests across 7 test classes |

## Architecture (ASCII)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CALIBRATION FEEDBACK LOOPS                      │
│                                                                        │
│  Reason engine produces confidence estimates                           │
│  ↓                                                                     │
│  ConfidenceCalibrator records (predicted, actual) pairs                │
│  ↓ computes ECE, bias, overconfidence flag                             │
│  ┌────────────────┬────────────────┬─────────────────────┐             │
│  │    νοῦς (iter-007)  │  πρᾶξις (iter-006/009)  │  μνήμη (iter-009) │
│  │                  │                  │                     │         │
│  │ Reasoning depth  │ Safety gates     │ Consolidation       │         │
│  │  D⨯amplifier     │  τ = 0.5+auto... │  threshold Δ_cal    │         │
│  │ Branching factor │ Verify depth     │  Δ = ECE × 0.30     │         │
│  │ Divergence force │  ω += clamp(...) │  + bonus_if_ovrcf   │         │
│  │                  │                  │                     │         │
│  │ Attention(iter-008)  │                  │ Pruning discount     │         │
│  │ retention mod    │                  │  λ = 1-min(0.3, ...) │         │
│  │ decay mod        │                  │                     │         │
│  │ GC threshold ↑   │                  │                     │         │
│  └────────────────┴────────────────┴─────────────────────┘             │
│                         ALL THREE CLOSED ✅                            │
└────────────────────────────────────────────────────────────────────────┘
```

## Core Mechanisms (with mathematical formulas)

### Mechanism 1: Calibration-Modulated Consolidation Thresholds

```python
# In HierarchicalMneme._modulate_thresholds():
Δ_cal = ECE × 0.30
if overconfident:
    Δ_cal += ECE × 0.20  # Overconfidence bonus — +50% more caution

cons_thresh = baseline + tension_mod + Δ_cal
prom_thresh = baseline + tension_mod + Δ_cal × 0.75
clamped to [0.15, 0.95] / [0.30, 0.98]
```

**Examples:**
- ECE = 0.02 (well-calibrated): Δ = 0.006 + 0 = 0.006 → negligible
- ECE = 0.15 (moderately miscalibrated): Δ = 0.045 (non-overconfident) or 0.075 (overconfident) → mild effect
- ECE = 0.25 (poorly calibrated, overconfident): Δ = 0.125 → significants shift push from 0.50 → 0.625
- ECE = 0.40 (extremely poor): Δ = 0.20 → 0.50 → 0.70, promotion 0.70 → 0.85

Requires ≥10 predictions to activate (avoids startup noise).

### Mechanism 2: Calibration-Aware Pruning Discount

```python
# During Phase 4 of consolidate():
if overconfident and total_predictions >= 10:
    discount = min(0.30, ECE × 2.0)          # Up to 30% reduction
    pruned = int(raw_pruned × (1.0 - discount))
    spared = raw_pruned - pruned              # Reprieved entries
    # Boost surviving entries just above forget threshold
    for each spared:
        strength = FORGET_THRESHOLD × 1.5     # Second chance
```

When the agent is overconfident, it systematically underestimates how valuable borderline memories may be. The pruning discount prevents overconfident memory deletion — entries near the forget threshold get a strength reprieve instead of immediate deletion.

### Mechanism 3: Cross-Pillar Calibration Push

```
CognitionPillar.update_tension_profile(profile):
    reasoning.set_tension_profile(profile)          # Always done
    attention.set_calibration_state(ECE, bias, ...)  # iter-008
    mneme_pillar.update_calibration(ECE, bias, ...)  # iter-009 ← NEW
```

The push is automatic — any agent loop that calls `update_tension_profile()` already triggers this. Zero migration cost for existing code.

### Mechanism 4: ConsolidationReport Calibration Fields

Three new fields on `ConsolidationReport`:
- `calibration_ece: float` — ECE value at consolidation time (0.0 when inactive)
- `calibration_active: bool` — whether calibration modulated this cycle
- `calibration_prune_saved: int` — how many entries spared from overconfident pruning

The `summary()` method appends `cal: ECE=0.XXX saved=N` when calibration is active.

## Tension Modulation

Calibration modulation is **additive** with existing tension-based modulation:

| Source | Effect on cons_threshold | Effect on prom_threshold |
|--------|--------------------------|--------------------------|
| consolidate_prune = -0.5 | −0.10 (more consolidation) | −0.075 |
| consolidate_prune = +0.5 | +0.10 (less consolidation) | +0.075 |
| specific_general > 0 | −0.05 (more abstraction) | −0.05 |
| ECE = 0.25 (not overconfident) | +0.075 (cautious) | +0.056 |
| ECE = 0.25 (overconfident) | +0.125 (very cautious) | +0.094 |

Final = baseline + tension_mod + calibration_mod, clamped to bounds.

## Test Coverage

| Test Class | Tests | What It Covers |
|-----------|-------|----------------|
| `TestCalibrationState` | 4 | Default state, round-trip, clear, independence from tensions |
| `TestCalibrationThresholdModulation` | 6 | No-cal, <10 guard, high-ECE, low-ECE, composition with tension, bounds |
| `TestConsolidationReportCalibration` | 5 | Default fields, custom values, active consolidation, inactive, summary format |
| `TestCalibrationPruneSensitivity` | 2 | Overconfident reduces prune, well-calibrated no discount |
| `TestMnemePillarCalibration` | 3 | Forwarding, <10 guard, uninitialized no-op |
| `TestCognitionPillarMnemePush` | 3 | Full push, no-error without mneme, <10 guard |
| `TestEndToEndCalibrationMneme` | 3 | Overconfident blocks low-sig, well-calibrated normal, pillar pipeline |
| **Total** | **26** | Full coverage of all calibration-aware paths and edge cases |

## Design Decisions

1. **Additive, not multiplicative, composition with tensions** — Matches the pattern established in attention (iter-008). Multiplicative composition can produce pathological oscillations at extreme values.

2. **≥10 prediction guard** — Consistent with calibrator (iter-005), reasoning amplifier (iter-007), and attention calibration (iter-008). Prevents startup noise from distorting early consolidation decisions.

3. **Overconfidence bonus (1.5× modulation)** — Overconfidence is systematically more dangerous than random miscalibration because it creates a consistent bias toward discarding genuinely useful information. The bonus (ECE × 0.20 vs base ECE × 0.30 = 1.66×) matches the 1.15× overconfidence bonus in reasoning (iter-007) and the 1.2× bonus in attention (iter-008), calibrated proportionally.

4. **Pruning discount capped at 30%** — Prevents pathological memory accumulation even from extreme overconfidence. A 30% reduction is enough to protect against systematic deletion without paralyzing maintenance.

5. **Pruning discount only applies to overconfident, not underconfident** — Underconfident agents are already too cautious; a pruning discount would compound excessive retention.

6. **Calibration-modulated promotion has 0.75× multiplier** — Promotion to semantic tier is a higher-stakes decision than consolidation to episodic; the calibration effect is proportionally smaller to avoid over-correcting.

7. **MnemePillar.update_calibration() is separate from update_tension_profile()** — Clean separation of concerns. Tension and calibration are orthogonal inputs to the memory system.

## Why This Creates Impact

### Short-term (immediate value)
- **Closes the final calibration loop** — all three pillars now respond to metacognitive quality. The framework is now *fully* calibration-aware at the pillar level.
- **Prevents garbage memory when the agent is confused** — an overconfident agent with poor calibration will not aggressively consolidate noise or prune useful context.
- **Drop-in upgrade** — existing code calling `update_tension_profile()` automatically benefits. Zero API changes required.

### Long-term (strategic value)
- **Foundation for cross-session calibration persistence** — now that Mneme stores calibration state, future iterations can persist calibrator data across sessions (the calibration state is already in the ConsolidationReport).
- **Enables calibration-gated delegation** — when ECE exceeds a threshold AND the mneme system is resisting consolidation, the agent has converging evidence it cannot trust its own judgment, creating a natural delegation trigger.
- **Completes the metacognitive architecture** — the pattern is now established across all three pillars: calibrator → pillar interface → tension-composed modulation → reportable metrics. This unified pattern enables future work like calibration-aware scheduling, rehearsal prioritization, and episodic→semantic abstraction quality.

## Architecture: Full Framework State

```
                             ╔══════════════════════╗
                             ║   EquilibriumEngine  ║
                             ║   (8 tension axes)   ║
                             ╚═════╤════════╤══════╝
                                   │        │
                    ┌──────────────┘        └──────────────┐
                    ▼                                      ▼
          ╔══════════════════════╗              ╔══════════════════════╗
          ║   Cognition (νοῦς)   ║◄──signals──►║   Praxis (πρᾶξις)   ║
          ╠══════════════════════╣              ╠══════════════════════╣
          ║ Attention System     ║              ║ ActionOrchestrator   ║
          ║  • Budget mgmt       ║              ║  • DAG scheduling    ║
          ║  • Keep/prune thr.   ║              ║  • Safety gates      ║
          ║  • Recency decay     ║              ║  • Calibrated verify ║
          ║  • GC cycles         ║              ║  • Parallel exec       ║
          ║  • Calib retention◄──╫────cal──────╫──→ Calib verify depth ║
          ║  • Calib decay mod   ║              ║                      ║
          ║ RecursiveReasoning   ║              ║                      ║
          ║  • Hyp decomposition ║              ║                      ║
          ║  • ConfidenceCalibrtr║──calibrator──╫──→                    ║
          ║  • Calib amplifier   ║              ║                      ║
          ╚════════════╤═════════╝              ╚══════════╤═══════════╝
                       │                                   │
                context│                                   │signals
                       │                                   │
                       ▼                                   ▼
          ╔══════════════════════╗              ╔══════════════════════╗
          ║      Mneme (μνήμη)   ║◄────────────║   Cross-Pillar       ║
          ╠══════════════════════╣  store/recall║   Signal Router      ║
          ║ HierarchicalMemory   ║              ╚══════════════════════╝
          ║  • Working (7±2)     ║
          ║  • Episodic (1K)     ║
          ║  • Semantic (10K)    ║
          ║  • Ebbinghaus decay  ║
          ║  • Spaced repetition ║
          ║  • Calib threshold ◄─╫──cal from cognition (iter-009 ⭐)
          ║  • Calib pruning disc║
          ╚══════════════════════╝
```

## Next Iteration Candidates

1. **Calibration persistence across sessions** — `MnemePillar.shutdown()` already serializes memory state. Extend `to_dict()`/`from_dict()` to include calibrator state so the agent resumes with accumulated calibration knowledge across restarts.

2. **Calibration-gated delegation** — When ECE exceeds a threshold (e.g., > 0.20) and the system has exhausted its calibration-aware correction mechanisms, delegate the current task to a subagent or request human input. This is the natural "I don't know what I don't know" guard.

3. **Rehearsal prioritization by calibration confidence** — When well-calibrated, rehearse significance-ranked entries. When poorly calibrated, rehearse more uniformly across tiers (don't skip "boring" entries the overconfident system may undervalue).

4. **Episodic→Semantic abstraction quality** — The pattern-support check for semantic promotion (`_has_pattern_support`) could also be calibration-aware: at high ECE, require more pattern support (more frequent bigrams/trigrams) before promoting to semantic.

## Files Changed (tree)

```
isonome/
├── cognition/
│   └── pillar.py            +19 lines — mneme_pillar ref + calibration push
└── mneme/
    ├── hierarchical.py      +128/−3 lines — core calibration mechanisms
    └── pillar.py            +24 lines — update_calibration() wiring
tests/
└── test_calibration_mneme.py  +604 lines — 26 tests
```

## Commit Stack

```
4ecc4b0 test: 26 tests for calibration-aware Mneme consolidation
6908df5 feat: CognitionPillar pushes calibration to MnemePillar each tick
86da023 feat: MnemePillar.update_calibration() — wire calibration push from Cognition
76134d2 feat: calibration-aware Mneme consolidation and pruning
a04d428 feat: calibration-driven verification depth in Praxis orchestrator    ⬅ prev iter
```
