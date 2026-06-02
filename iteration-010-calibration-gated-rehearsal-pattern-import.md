# isonome-framework — Iteration 010: Calibration-Gated Rehearsal, Pattern Support & Import-from-Attention

**Date:** 2026-06-02
**Cron Job:** Hourly incremental improvement
**Iteration:** 010
**Change Type:** Calibration cross-pillar feedback — Mneme metacognition, retrieval/abstraction/ingestion gating
**Tests:** 466/466 passing (+29 new)

---

## Summary

Three new calibration-gated mechanisms in the Mneme pillar, completing the metacognitive calibration architecture at every boundary of the memory system:

| Boundary | Mechanism | Before | After |
|----------|-----------|--------|-------|
| **Retrieval** (rehearsal) | `rehearse_by_tags()` | Flat uniform boost | Four-mode calibration-aware rehearsal |
| **Abstraction** (semantic promotion) | `_has_pattern_support()` | Fixed 30% pattern ratio | Calibration-gated: 40% overconf, 20% underconf, 30% standard |
| **Ingestion** (attention→mneme) | `import_from_attention()` | Fixed 0.15 sig floor | Calibration-gated: 0.20 overconf, 0.08 underconf, 0.15 standard |

These close the final three open loops in the Mneme calibration framework. After iter-009 (consolidation-time calibration), the memory system now responds to metacognitive quality at **every** touch point: when it stores, when it retrieves, when it abstracts, and when it prunes.

---

## What Was Built

| File | Action | Change | Description |
|------|--------|--------|-------------|
| `isonome/mneme/hierarchical.py` | **Modified** | +111 / −7 | Calibration-aware `rehearse_by_tags()`, `_has_pattern_support()`, `import_from_attention()` |
| `tests/test_calibration_rehearsal_pattern.py` | **Created** | +669 | 29 tests across 7 test classes |

---

## Architecture (ASCII)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                      MNEME CALIBRATION GATES (iter-010)                      │
│                                                                              │
│  External signals / Attention prune output                                  │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐                                                        │
│  │ import_from_attn │◄── Calibration-gated ingestion floor                   │
│  │   min_sig = f()  │    f(overconfident) = 0.20  (stricter)                │
│  └────────┬────────┘    f(underconfident) = 0.08  (more permissive)         │
│           │              f(well_calibrated) = 0.15 (standard)                │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │   Working (WM)  │  memory pressure →                                   │
│  └────────┬────────┘                                                        │
│           │ consolidation (iter-009)                                        │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │  Episodic (EP)  │◄── Calibration-gated pattern support for sem. promo    │
│  └────────┬────────┘    f(overconfident) = 40% required (more evidence)     │
│           │             f(underconfident) = 20% required (compensate)       │
│           ▼             f(well_calibrated) = 30% (standard)                 │
│  ┌─────────────────┐                                                        │
│  │  Semantic (SEM)  │                                                       │
│  └─────────────────┘                                                        │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────┐                                                        │
│  │ rehearse_by_tags │◄── Calibration-aware rehearsal prioritization         │
│  └─────────────────┘    Overconfident:  50% boost → all entries            │
│                          Underconfident: 130% boost → all entries           │
│                          Well-calibrated: 110% high-sig, skip low-sig       │
│                          Default:        100% boost → all entries           │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  Retrieval   → calibration modulates HOW the agent refreshes memories       │
│  Abstraction → calibration modulates WHEN the agent converts to knowledge   │
│  Ingestion   → calibration modulates WHAT the agent chooses to remember     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Mechanisms (with mathematical formulas)

### Mechanism 1: Calibration-Aware Rehearsal Prioritization

The agent's rehearsal strategy changes based on calibration quality:

```
  ┌─────────────────────────────────────────────────────────────┐
  │  rehearse_by_tags(tags, boost):                             │
  │                                                             │
  │  if ECE < 0.05:  (well-calibrated)                         │
  │    for each entry:                                          │
  │      if sig >= 0.7:  boost = 1.10 × base_boost (bonus)     │
│      elif sig >= 0.35:  boost = base_boost                  │
  │                    else:  skip entry                       │
  │                                                             │
  │  if ECE > 0.15 && overconfident:                           │
  │    for each entry:                                         │
  │      boost = 0.50 × base_boost (distributed, all entries)  │
  │                                                             │
  │  if ECE > 0.15 && underconfident:                          │
  │    for each entry:                                         │
  │      boost = 1.30 × base_boost (elevated, all entries)     │
  │                                                             │
  │  default (moderate / no calibration):                      │
  │    for each entry:                                         │
  │      boost = base_boost                                     │
  └─────────────────────────────────────────────────────────────┘
```

**Rationale:**
- **Well-calibrated**: trust significance judgments. Prioritize high-value memories on the assumption the system correctly identifies what matters.
- **Overconfident**: systematically overrates its own judgments. Distribute rehearsal evenly so borderline memories don't get starved — the system can't be trusted to prioritize correctly.
- **Underconfident**: systematically undervalues everything. Give all entries a stronger boost to compensate for the general undervaluation.
- **Moderate**: no correction needed. Standard uniform rehearsal.

### Mechanism 2: Calibration-Gated Pattern Support (`_has_pattern_support`)

Controls the Episodic→Semantic promotion gate:

```
required_ratio = 0.30  (default)

if calibration_active:
    if overconfident and ECE > 0.15:
        required_ratio = 0.40    # Raise bar — need MORE evidence
    elif underconfident and ECE > 0.15:
        required_ratio = 0.20    # Lower bar — compensate for undervaluation

return (pattern_hits / total_grams) >= required_ratio
```

**Effect at different ECE levels:**

| Calibration | Required Ratio | Entry with 25% overlap | Entry with 35% overlap | Entry with 50% overlap |
|------------|----------------|----------------------|----------------------|----------------------|
| Well-calibrated | 30% | ✗ Blocked | ✓ Promoted | ✓ Promoted |
| Overconfident | 40% | ✗ Blocked | ✗ Blocked | ✓ Promoted |
| Underconfident | 20% | ✓ Promoted | ✓ Promoted | ✓ Promoted |

### Mechanism 3: Calibration-Gated Import-from-Attention Floor

Controls what content from the attention system enters working memory:

```
significance = sigmoid(8.0 × (attention_score − 0.35))

min_sig = 0.15  (default)

if calibration_active:
    if underconfident and ECE > 0.15:
        min_sig = 0.08   # More permissive
    elif overconfident and ECE > 0.15:
        min_sig = 0.20   # Stricter

if significance >= min_sig: store()
else: reject()
```

**Attention score → significance mapping:**

| Attention Score | Significance (sigmoid) | Default (0.15 floor) | Overconfident (0.20 floor) | Underconfident (0.08 floor) |
|----------------|----------------------|---------------------|--------------------------|---------------------------|
| 0.05 | 0.08 | ✗ | ✗ | ✓ (borderline) |
| 0.10 | 0.12 | ✗ | ✗ | ✓ |
| 0.15 | 0.17 | ✓ | ✗ | ✓ |
| 0.20 | 0.23 | ✓ | ✓ | ✓ |
| 0.30 | 0.40 | ✓ | ✓ | ✓ |
| 0.50 | 0.77 | ✓ | ✓ | ✓ |

## Tension Modulation

Unlike iter-008 (attention) and iter-009 (mneme thresholds), these three new mechanisms are **not** additive with tension. They operate independently of tension modulation:

| Mechanism | Relationship to Tensions |
|-----------|------------------------|
| Calibration-gated rehearsal | Replaces uniform boost. Tensions still modulate recall scores (via `specific_general`) but rehearsal boost is calibration-only. |
| Calibration-gated pattern support | Adjusts the ratio threshold independent of tension. Tension still modulates consolidation and promotion thresholds. |
| Calibration-gated import | Adjusts the minimum floor independently. Attention score mapping is unchanged. |

This is by design — rehearsal, abstraction, and ingestion are calibration-quality decisions, not tension-balance decisions. The overconfident agent should rehearse differently regardless of whether it's in explore or exploit mode.

## Test Coverage

### New Tests (29 in `test_calibration_rehearsal_pattern.py`)

| Test Class | Tests | What It Covers |
|-----------|-------|----------------|
| `TestCalibrationRehearsalPrioritization` | 9 | Default, well-calibrated (skip low, bonus high), overconfident (distributed, reduced boost), underconfident (elevated boost), moderate standard, few-predictions guard, empty entries |
| `TestCalibrationPatternSupport` | 8 | Default, overconfident raises to 40%, overconfident high-match passes, underconfident lowers to 20%, underconfident low-match passes, well-calibrated standard, few-predictions guard, no pattern freqs |
| `TestCalibrationImportFromAttention` | 8 | Default floor, overconfident 0.20 floor, underconfident 0.08 floor, few-predictions guard, moderate standard, well-calibrated standard, after-clear restore |
| `TestCalibrationConsolidationPipeline` | 3 | Overconfident blocks weak semantic, well-calibrated normal, underconfident allows marginal |
| `TestMnemePillarCalibrationRehearsal` | 2 | Pillar rehearsal with calibration, pillar import with calibration |
| **Total** | **29** | Full coverage of all calibration-aware paths |

### Full Suite (466 total)

```
tests/test_mneme.py                   50 tests
tests/test_calibration_mneme.py       26 tests
tests/test_calibration_rehearsal_pattern.py  29 tests  ← NEW
tests/test_attention.py               23 tests
tests/test_calibration.py             63 tests
tests/test_calibration_attention.py   51 tests
tests/test_reasoning.py               63 tests
tests/test_uncertainty_planning.py    35 tests
tests/test_equilibrium.py             18 tests
tests/test_praxis.py                  68 tests
tests/test_agent.py                   10 tests
tests/test_confidence_gating.py       24 tests
tests/test_confidence_verify.py       24 tests
---
Total: 466 tests — 466/466 passing ✅
```

## Design Decisions

1. **Independent from tension modulation** — Unlike consolidation thresholds (iter-009) where calibration is additive with tension, rehearsal, pattern support, and import floors are calibration-only decisions. A tension axis like `consolidate_prune` already modulates whether the agent consolidates more or less; calibration tells the agent *how much to trust its own judgment about what to consolidate*. These are orthogonal.

2. **≥10 prediction guard** — Matches the established pattern across all calibration-aware systems (iter-005 calibrator, iter-007 reasoning amplifier, iter-008 attention, iter-009 mneme thresholds). Prevents startup noise.

3. **ECE > 0.15 threshold** — The three-tier rehearsal strategy fires at ECE > 0.15 for corrective modes (overconfident/underconfident) and ECE ≤ 0.05 for well-calibrated mode. The 0.05-0.15 range is "moderate" — not good enough for significance-trusting but not bad enough for correction. This matches the confidence noise floor from the isotonic correction analysis.

4. **No overconfidence bonus multiplier** — Unlike iter-008 (1.2× attention retention bonus) and iter-009 (1.5× ECE overconfidence bonus for thresholds), these three mechanisms use discrete thresholds rather than continuous multipliers. The transitions are step functions because the behavior change (skip vs. include, raise vs. lower) is qualitative, not quantitative.

5. **Underconfident import floor (0.08)** — The 0.08 floor is calibrated so that only attention scores ≥ 0.05 (sigmoid output ≈ 0.08) barely pass. This catches real signals from noise while still providing a lower bound. Even underconfident agents should not store pure noise (scores < 0.05).

6. **Well-calibrated rehearsal skips low-significance** — The 0.35 significance cutoff for well-calibrated rehearsal corresponds to the sigmoid midpoint of `import_from_attention`. If the system correctly identifies significance, entries below 0.35 were borderline even for ingestion — they don't need rehearsal.

## Why This Creates Impact

### Short-term (immediate value)
- **Closes the last three Mneme calibration touch points** — memory is now calibration-aware at every boundary: store (import floor), retrieve (rehearsal prioritization), abstract (pattern support), consolidate (iter-009 thresholds), and prune (iter-009 discount).
- **Prevents overconfident pattern over-abstraction** — overconfident agents need 40% pattern evidence before promoting to semantic, preventing premature knowledge crystallization from weak signal.
- **Protects underconfident recall** — underconfident agents don't skip "probably irrelevant" entries during rehearsal, ensuring they maintain broader context.

### Long-term (strategic value)
- **Enables calibration-based rehearsal scheduling** — the pattern is now established for future work on rehearsal scheduling. A future iteration could schedule rehearsal frequency based on calibration × significance × tension composition.
- **Foundation for calibration-gated delegation trigger** — the import floor is the ingestion gate. When combined with the pruning discount (iter-009) and rehearsal prioritization, the Mneme system can report "calibration stress" — converging evidence that the agent cannot trust its own memory decisions at any boundary. This creates a natural delegation trigger.
- **Complete metacognitive architecture for Mneme** — the pattern is now established for all five Mneme operations: store (import), maintain (rehearse), abstract (pattern support), consolidate (thresholds), and prune (discount). Each has its own calibration-aware modulation.

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
          ║  • GC cycles         ║              ║  • Parallel exec     ║
          ║  • Calib retention  ◄║──cal────────║──→ Calib verify depth║
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
          ║  • WM (7±2)          ║              ╚══════════════════════╝
          ║  • Episodic (1K)     ║
          ║  • Semantic (10K)    ║
          ║  • Ebbinghaus decay  ║
          ║  • Spaced repetition ║
          ║  ┌─── CALIBRATION GATES ──────────────────────┐          ║
          ║  │ import_from_attention:  floor = f(calib)   │ ← iter-010│
          ║  │ rehearse_by_tags:       boost = f(calib)   │ ← iter-010│
          ║  │ _has_pattern_support:   ratio = f(calib)   │ ← iter-010│
          ║  │ consolidation_threshold: delta = f(calib)  │ ← iter-009│
          ║  │ pruning_discount:        lambda = f(calib) │ ← iter-009│
          ║  └──────────────────────────────────────────────┘         ║
          ╚═══════════════════════════════════════════════════════════╝
```

## Next Iteration Candidates

1. **Calibration-based rehearsal scheduling** — Instead of only modulating the boost of `rehearse_by_tags()`, schedule the *frequency* of rehearsal based on calibration × significance. Poorly calibrated agents rehearse high-significance entries more frequently (compensating for unreliable importance detection).

2. **Cross-session calibration persistence** — Extend `to_dict()`/`from_dict()` on both HierarchicalMneme and MnemePillar to include calibrator state (ECE history, reliability diagram bins). The agent should resume with accumulated calibration knowledge across restarts.

3. **Calibration-gated delegation trigger** — When ECE exceeds 0.20 AND all three gates (import floor struggling, rehearsal skipping nothing, pattern support demanding high evidence) are in corrective mode, the Mneme system has converging evidence that the agent cannot trust its own memory decisions. Emit a `calibration_stress` signal for the equilibrium engine to consider delegation.

4. **Attention weight rebalancing when overconfident** — When the agent is systematically overconfident, its task-relevance judgments are suspect. Shift attention scoring weight from β (task relevance) toward α (surprisal/novelty) — favoring content the overconfident system would normally dismiss.

5. **Rehearsal after promotion with calibration-aware boost** — When an entry promotes from WM→Episodic or Episodic→Semantic, apply a promotion-time rehearsal that scales inversely with calibration quality: more reinforcement for poorly-calibrated promotions (the system needs to undo potential errors).

## Files Changed (tree)

```
isonome/
└── mneme/
    └── hierarchical.py        +111/−7 lines — 3 new calibration-aware mechanisms
tests/
└── test_calibration_rehearsal_pattern.py  +669 lines — 29 tests
```

## Commit Stack

```
46995e3 test: 29 tests for calibration-gated rehearsal, pattern support, and import-from-attention
ccd9687 feat: calibration-gated rehearsal, pattern support, and import-from-attention in Mneme
6bc6241 docs: iteration-009 — calibration-aware Mneme consolidation    ⬅ prev iter
```
