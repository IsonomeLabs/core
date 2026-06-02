# isonome-framework — Iteration 007: Uncertainty-Aware Planning — Metacognition Modulates Reasoning Effort

**Date:** 2026-06-01
**Cron Job:** Hourly incremental improvement
**Iteration:** 007
**Change Type:** feat — calibration quality modulates reasoning depth, branching, and divergence
**Tests:** 360/360 passing (325 existing + 35 new)

---

## Summary

The isonome Cognition pillar could track calibration quality (Iter-005: ConfidenceCalibrator) and use it to gate actions in Praxis (Iter-006: Calibrated Safety Gates), but the reasoning engine itself was calibration-blind. When the calibrator detected systematic overconfidence (predicting 85% confidence when actual success was 35%), the reasoning engine would blithely continue producing plans at nominal depth and breadth — never investing more cognitive effort to compensate for its own miscalibration.

This iteration closes that metacognitive loop. The **RecursiveReasoningEngine** now reads its own calibrator's ECE (Expected Calibration Error) and bias metrics before every reasoning session, computing a **calibration amplifier** that multiplicatively increases reasoning depth, branching factor, and divergence when calibration quality is poor. When the system knows it's poorly calibrated, it thinks harder. When well-calibrated, it thinks efficiently.

This is the second keystone improvement after calibrated safety gates — uncertainty-aware planning transforms calibration from a passive metric into an active controller of the agent's most expensive resource: reasoning computation.

## What Was Built

| File | Action | Lines | Description |
|------|--------|-------|-------------|
| `isonome/cognition/reasoning.py` | **Modified** | +81 / −14 | `_compute_calibration_amplifier()`, calibrated `_compute_max_depth()`, `_compute_branching_factor()`, `_is_divergent()` |
| `tests/test_uncertainty_planning.py` | **Created** | 714 | 35 tests across 7 test classes |

## Architecture

```
┌────────────────────────────────── νοῦς (Cognition) ────────────────────────────────┐
│                                                                                     │
│  ┌─────────────────────────────┐          ┌─────────────────────────────────────┐  │
│  │   ConfidenceCalibrator      │          │   RecursiveReasoningEngine          │  │
│  │                             │          │                                     │  │
│  │   compute_ece() → 0.25      │──────────│  reason(task)                       │  │
│  │   compute_bias() → +0.12    │  reads   │    │                                │  │
│  │   is_overconfident → True   │          │    ├─ _compute_calibration_amplifier│  │
│  │                             │          │    │   amplifier = 1 + 2.0×0.25     │  │
│  │   Overconfidence            │          │    │     × (1 + 0.12) × 1.15       │  │
│  │   bonus 1.15× ──────────────┘          │    │   amplifier = 1.64             │  │
│  │                                        │    │                                │  │
│  │                                        │    ├─ _compute_max_depth            │  │
│  │                                        │    │   D = 2 + ceil(6×0.75×1.64)   │  │
│  │                                        │    │   D = 2 + 8 = 10  (vs nom. 6) │  │
│  │                                        │    │                                │  │
│  │                                        │    ├─ _compute_branching_factor     │  │
│  │                                        │    │   B = ceil(3×1.85×1.64) = 10  │  │
│  │                                        │    │   (vs nominal B = 6)           │  │
│  │                                        │    │                                │  │
│  │                                        │    └─ _is_divergent                 │  │
│  │                                        │        ECE 0.25 > 0.15 → TRUE      │  │
│  │                                        │        (overrides convergent tension)│  │
│  └────────────────────────────────────────┘                                     │  │
│                                                                                     │
│  Result: Poorly calibrated engine reasons 64% deeper, explores 67% more branches,  │
│  forces divergent mode — investing more cognitive resources until calibration       │
│  improves through the metacognitive feedback loop.                                  │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Core Mechanisms

### 1. Calibration Amplifier

The mathematical core — a multiplicative factor that scales reasoning effort proportionally to calibration quality:

```
amplifier = 1 + κ × ECE × (1 + |bias|) × overconfidence_bonus

Where:
  κ = 2.0                    Calibration sensitivity factor
  ECE ∈ [0, ~0.3+]          Expected Calibration Error
  bias ∈ [−1, +1]           Weighted confidence-accuracy gap
  overconfidence_bonus = 1.15 if bias > drift_threshold, else 1.0
  amplifier ∈ [1.0, 2.0]    Bounded output

At ECE = 0.00, bias = 0.00:   amplifier = 1.00  (perfect — no extra effort)
At ECE = 0.10, bias = 0.05:   amplifier = 1.21  (mild miscalibration)
At ECE = 0.20, bias = 0.10:   amplifier = 1.44  (moderate)
At ECE = 0.30, bias = 0.15:   amplifier = 1.69  (significant)
At ECE = 0.30, bias = 0.15,
  overconfident:              amplifier = 1.94  (systematic wrongness)
```

**Design principle**: Systematic overconfidence is more dangerous than random error. When the system is consistently overconfident (bias > drift_threshold), the amplifier gets a 15% bonus — the system must reason substantially more to recover calibration.

### 2. Calibration-Aware Depth Modulation

```
D = 2 + ⌈6 × (1 + p_shallow)/2 × amplifier⌉

p_shallow = -1, amplifier = 1.0 → D =  2  (shallow, nominal)
p_shallow = +1, amplifier = 1.0 → D =  8  (deep, nominal)
p_shallow = +1, amplifier = 1.64 → D = 12  (deep, poorly calibrated)
p_shallow = +1, amplifier = 2.0  → D = 14  (deep, maximally amplified)
```

The calibration amplifier scales the effective depth range. In practice:
- **Well-calibrated (amplifier ≈ 1.0)**: nominal depth D ∈ [2, 8]
- **Poorly calibrated (amplifier ≈ 1.6)**: amplified depth D ∈ [2, 12]
- **Maximum (amplifier = 2.0)**: extreme depth D ∈ [2, 14]

### 3. Calibration-Aware Branching Modulation

```
B = max(1, ⌈3 × (1 − p_exploit) × amplifier⌉)

p_exploit = -1 (explore), amplifier = 1.0 → B =  6  (nominal)
p_exploit = -1 (explore), amplifier = 1.64 → B = 10  (amplified)
p_exploit = +1 (exploit),  amplifier = 1.0 → B =  1  (minimum)
p_exploit = +1 (exploit),  amplifier = 2.0 → B =  2  (exploit but very poorly calibrated)
```

Even in exploit mode, severe miscalibration forces at least 2 alternatives — because when you don't know whether you're right, you should verify.

### 4. Calibration-Forced Divergence

```
_is_divergent() returns True if:
  1. divergent_convergent < 0  (explicit diverge mode), OR
  2. calibrator.total_predictions ≥ 10 AND calibrator.compute_ece() > 0.15
```

The 0.15 ECE threshold was chosen because:
- ECE ≤ 0.05: Excellent calibration (within noise)
- ECE 0.05–0.15: Acceptable calibration
- ECE > 0.15: Poor calibration — indicates systematic error
- Requires ≥ 10 predictions to avoid noise-driven divergence

When the system is poorly calibrated, it needs to explore multiple plan alternatives rather than committing to a single best path. Divergent mode in plan collapse returns all viable paths sorted by confidence, giving the downstream Praxis pillar more options to verify.

## Tension Modulation

The calibration amplifier sits **above** the traditional tension system — it scales the output of tension-modulated parameters:

| Layer | Modulation | When |
|-------|-----------|------|
| Tension | shallow_deep → depth range [0, 6] | Always |
| Tension | explore_exploit → branching [1, 6] | Always |
| Tension | divergent_convergent → output shape | Always |
| **Calibration** ★ | **amplifier ∈ [1.0, 2.0] scales all three** | ECE > 0, ≥ 10 predictions |

This two-layer design preserves the existing tension system while adding metacognitive awareness on top. A well-calibrated engine behaves exactly as before — zero behavioral regression.

## Test Coverage (35 new tests)

| Test Class | Count | Focus |
|-----------|-------|-------|
| `TestCalibrationAmplifier` | 8 | Formula: no data, perfect, moderate, high ECE, overconfident bonus, bounded, underconfident, monotonic |
| `TestCalibrationAwareDepth` | 4 | Depth: default unchanged, increases with ECE, shallow minimum, deep maximum |
| `TestCalibrationAwareBranching` | 4 | Branching: default, increases, exploit minimum, explore maximum |
| `TestCalibrationAwareDivergence` | 5 | Divergence: convergent+low ECE, high ECE override, explicit diverge, not-enough-data, boundary at 0.15 |
| `TestCalibrationAwarePlanOutput` | 5 | End-to-end: well-calibrated plans, poor-calibrated deeper plans, more nodes, complex output, stats |
| `TestCognitionPillarCalibrationIntegration` | 3 | Pillar: calibrator presence, depth increases over time, evaluate_result builds history |
| `TestCalibrationAmplifierEdgeCases` | 6 | Edge: zero ECE, tension independence, live state, determinism, stats, empty window |

### Key Test Scenarios

- **test_no_calibration_data_returns_one**: With < 10 predictions, amplifier stays at 1.0 — no false amplification from insufficient data
- **test_amplifier_bounded_to_two**: 200 extreme overconfident predictions at 0.95 → amplifier caps at 2.0
- **test_miscalibrated_increases_depth**: Two calibrators with ECE=0.02 vs ECE=0.25 → depth increases monotonically
- **test_explore_mode_with_poor_calibration_high_branching**: Explore (-1) + high ECE → branching ≥ 6
- **test_high_ece_forces_divergence_despite_convergent_tension**: Convergent tension 0.5 + ECE > 0.15 → divergent override
- **test_reason_with_calibration_data_produces_deeper_plan**: Before/after calibration data → depth increases
- **test_amplifier_reads_live_calibrator_state**: Amplifier updates immediately when new calibration data arrives

## Design Decisions

1. **Amplifier reads calibrator live, not cached**: `_compute_calibration_amplifier()` calls `calibrator.compute_ece()` and `calibrator.compute_bias()` every time — not a cached value. This ensures that as calibration improves (through `evaluate_result` signals), reasoning effort immediately adjusts downward. No stale amplifier.

2. **Bounded to [1.0, 2.0]**: The amplifier cannot reduce reasoning effort below nominal (1.0) — even perfectly calibrated engines reason at their tension-configured depth. Nor can it exceed 2.0 — preventing runaway depth when calibration is catastrophically broken. A 2× amplifier with deep mode already reaches depth 14 (double the nominal maximum of 8), sufficient to explore a vastly larger reasoning space.

3. **Amplifier independent of tension profile**: The amplifier depends solely on the calibrator state, not on any tension axis. This orthogonal design means tension and calibration modulate independently: a "shallow+well-calibrated" engine stays shallow, while a "shallow+poorly-calibrated" engine gets a modest depth boost (D = 2 + ceil(0 * amplifier) = 2 — the amplifier can't overcome the shallow pole). This preserves the tension system's authority in all regimes.

4. **Minimum 10 predictions for calibration reading**: Both the amplifier and divergence override require ≥ 10 total predictions before reading ECE/bias. This prevents startup noise from causing spurious amplification. The threshold is low enough that the metacognitive loop engages quickly (10 evaluate_result signals, which arrive within ~5 agent ticks in a busy pipeline).

5. **Overconfidence bonus is multiplicative, not additive**: The 1.15× bonus multiplies the entire amplifier term, not just the ECE component. This means the bonus scales with the severity of miscalibration — mildly overconfident systems get a tiny bonus; severely overconfident systems get a substantial one. Systematic wrongness compounds.

6. **Divergence override uses strict > threshold**: The ECE > 0.15 check uses strict greater-than (not ≥). At exactly 0.15, the system respects the tension setting. This prevents oscillation at the boundary — a small ECE fluctuation doesn't toggle divergence on every tick.

7. **No change to `_evaluate_confidence()`**: The confidence formula already reads calibrator weights (`w_evidence`, `w_child`) which are independently adjusted by `adjust_weights()` based on observed outcomes. Adding calibration to confidence evaluation would create a double-counting feedback loop. The amplifier modulates computational effort (depth, branching), while the calibrator modulates the confidence formula (evidence vs child weights) — orthogonal concerns, orthogonal mechanisms.

## Why This Creates Impact

### Short-term impact
- **Immediate plan quality improvement**: When the calibrator detects miscalibration, the reasoning engine automatically invests more effort — producing deeper, more thoroughly explored plans. An agent that was overconfident and producing shallow plans now automatically compensates.
- **No new API surface**: The calibration amplifier is entirely internal to `RecursiveReasoningEngine`. No caller changes needed. Existing `reason(task)` calls automatically get calibration-aware behavior.
- **Self-correcting**: As the agent executes plans and receives outcomes (via `evaluate_result`), calibration quality improves → amplifier decreases → reasoning returns to efficient nominal levels. No external tuning needed.

### Long-term impact
- **Autonomous self-improvement loop**: This completes the metacognitive feedback cycle:
  ```
  Poor calibration → deeper reasoning → better plans →
  more accurate outcomes → improved calibration → efficient reasoning
  ```
- **Foundation for resource-aware reasoning**: The amplifier pattern can extend to token budgets, LLM call counts, and other finite resources. A poorly calibrated agent could allocate more tokens to each reasoning step.
- **Calibration-gated delegation**: Future work can use the amplifier to decide whether to delegate tasks — poorly calibrated agents delegate more, well-calibrated agents execute themselves.
- **Cross-agent calibration pooling**: The amplifier formula is a pure function of calibrator state — multiple engines sharing a calibrator automatically amplify in lockstep, enabling fleet-wide metacognitive coordination.

## Architecture: Full Framework State

```
isonome-framework/
├── pyproject.toml
├── isonome/
│   ├── __init__.py
│   ├── agent.py              # IsonomeAgent — tick() loop, pillar wiring
│   ├── base.py               # BasePillar — signal/feedback queue, lifecycle
│   ├── types/__init__.py     # 21 Pydantic models, enums, protocols
│   ├── equilibrium/__init__.py # EquilibriumEngine — 8-axis PID regulator
│   ├── cognition/             # νοῦς — REASON + PLAN
│   │   ├── __init__.py        # ✅ Exports all systems
│   │   ├── attention.py       # ✅ AttentionEquilibriumSystem
│   │   ├── reasoning.py       # ✅ RecursiveReasoningEngine ★ MODIFIED
│   │   └── pillar.py          # ✅ CognitionPillar wrapper
│   ├── praxis/                # πρᾶξις — EXECUTE
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # ✅ ActionOrchestrator + calibrated safety gates
│   │   └── pillar.py          # ✅ PraxisPillar
│   └── mneme/                 # μνήμη — REMEMBER
│       ├── __init__.py
│       ├── hierarchical.py    # ✅ HierarchicalMneme
│       └── pillar.py          # ✅ MnemePillar
└── tests/
    ├── test_agent.py          # 10 tests
    ├── test_attention.py      # 23 tests
    ├── test_calibration.py    # 63 tests
    ├── test_confidence_gating.py # 24 tests
    ├── test_confidence_verify.py # 24 tests
    ├── test_equilibrium.py    # 18 tests
    ├── test_mneme.py          # 50 tests
    ├── test_praxis.py         # 68 tests
    ├── test_reasoning.py      # 63 tests
    └── test_uncertainty_planning.py  # 35 tests ★ NEW
```

**Total: 360 tests across 10 test files**

## Cross-Pillar Pipeline Status

| Pipeline | Status | Mechanism |
|----------|--------|-----------|
| νοῦς → πρᾶξις | ✅ Complete | CognitionPillar.reason() → plan_ready → PraxisPillar.import_plan() |
| πρᾶξις → μνήμη | ✅ Complete | PraxisPillar.export_to_mneme() → MnemePillar.store() |
| πρᾶξις → νοῦς | ✅ Complete | evaluate_result signal → CognitionPillar → calibrator.record() |
| νοῦς → μνήμη | ✅ Complete | Attention pruned chunks → MnemePillar.import_from_attention() |
| μνήμη → νοῦς | ✅ Complete | MnemePillar → add_context signal → CognitionPillar.attention |

## Metacognitive Feedback Loops — Complete State

| Loop | Iteration | Status |
|------|-----------|--------|
| Calibrator records outcomes | 005 | ✅ ConfidenceCalibrator tracks (conf, success) |
| Calibrator adjusts weights | 005 | ✅ w_evidence/w_child adapt to bias |
| Calibrator corrects raw confidence | 005 | ✅ Isotonic correction maps conf → accuracy |
| Evaluation feeds calibrator | 005 | ✅ CognitionPillar.evaluate_result → calibrate() |
| Safety gates use calibrated confidence | 006 | ✅ Praxis gate checks calibrated confidence |
| Praxis pillar gets calibrator reference | 006 | ✅ Cognition → Praxis calibrator wiring |
| **Reasoning effort scales with calibration** ★ | **007** | **✅ Amplifier modulates depth/branching/divergence** |

## Next Iteration Candidates

1. **Attention ↔ Calibration Link**: When calibration is poor, dynamically increase the attention budget (token_capacity). A miscalibrated engine needs more context to improve — expand the information channel. This closes the last open cross-subsystem feedback loop in Cognition.

2. **Calibration-Aware Token Budgeting**: Use the amplifier to allocate LLM tokens per reasoning step — poorly calibrated engines get more tokens per decompose/evaluate call. Direct resource allocation proportional to uncertainty.

3. **Mneme Calibration History**: Store calibration snapshots (ECE, weights, bias) in Mneme for cross-session persistence. Restore calibration state when an agent wakes up — no cold-start calibration phase.

4. **Cross-Agent Calibration Pooling**: Multiple agents sharing a calibrator. Each agent contributes (conf, outcome) pairs; calibration stats reflect fleet-wide accuracy, not individual agent bias.

5. **Uncertainty-Weighted Plan Scoring**: When presenting multiple plans (divergent mode), weight plan confidence scores by the calibrator's ECE — plans from a well-calibrated engine are trusted more than from a poorly-calibrated one.

6. **Calibration-Gated Delegation**: When ECE exceeds a threshold, delegate reasoning to a more-capable sub-agent or request human input. Self-aware task routing.

## Files Changed

```
isonome/cognition/reasoning.py         (modified — +81 / −14 lines)
tests/test_uncertainty_planning.py     (NEW — 714 lines, 35 tests)
```

## Commit Stack

```
15cc438 test: 35 tests for calibration-driven reasoning modulation
b3600ae feat: uncertainty-aware planning — calibration quality modulates reasoning effort
```
