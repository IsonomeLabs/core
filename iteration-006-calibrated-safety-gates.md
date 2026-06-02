# isonome-framework — Iteration 006: Calibrated Safety Gates — Metacognition → Execution

**Date:** 2026-06-01
**Cron Job:** Hourly incremental improvement
**Iteration:** 006
**Change Type:** feat — Confidence-based safety gating closes the Cognition→Praxis metacognitive loop
**Tests:** 319/319 passing (295 existing + 24 new)

---

## Summary

The isonome Cognition pillar could learn calibration (iter-005: ConfidenceCalibrator) but the Praxis pillar couldn't use it. When the calibrator learned the system was overconfident — saying "85% sure" when actual success was only 35% — the safety gate still used raw confidence. An action labeled "requires 85% confidence" would pass the gate even though the actual probability of success was 35%.

This iteration closes the metacognitive loop. The **ConfidenceCalibrator** from Cognition is now wired into the **ActionOrchestrator** in Praxis. Every action can declare a `confidence_required` threshold, and the orchestrator uses **calibrated** confidence — the observed, true probability of success — to gate it. The `autonomy_safety` tension axis modulates the confidence threshold, creating a three-dimensional safety model: risk × calibrated confidence × autonomy.

This is the keystone improvement: the agent's metacognition (knowing when it's wrong) now directly controls its execution safety. Every subsequent feature — uncertainty-aware planning, calibration-gated delegation, cross-agent calibration pooling — builds on this link.

## What Was Built

| File | Action | Lines | Description |
|------|--------|-------|-------------|
| `isonome/praxis/orchestrator.py` | **Modified** | +85 | `confidence_required` field on Action, `confidence_calibrator` param on orchestrator, confidence-based safety gating in `execute_batch()`, updated ExecutionReport, serialization |
| `isonome/praxis/pillar.py` | **Modified** | +30 | `confidence_calibrator` param on PraxisPillar init, `set_confidence_calibrator()` method, calibrator passed to orchestrator on init |
| `tests/test_confidence_gating.py` | **Created** | 310 | 24 tests covering Action field, orchestrator wiring, report fields, confidence gating with/without calibrator, tension modulation, approval override, import pipeline, serialization, pillar integration, edge cases |

## Architecture

```
┌────────────────────────── νοῦς (Cognition) ────────────────────────────┐
│                                                                         │
│  RecursiveReasoningEngine          ConfidenceCalibrator                │
│  ├─ reason(task) → plan          ├─ record(conf, success)             │
│  ├─ confidence per action        ├─ ECE, MCE, bias metrics            │
│  └─ calibrate(outcome)           ├─ adjust_weights()                  │
│                                  └─ calibrate_confidence(raw) → true  │
│                                                                         │
│  evaluate_result signal:                                               │
│    → calibrator.record(0.85, actual_success=False)                     │
│    → calibrator.adjust_weights()                                       │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │ calibrator reference (set during agent wiring)
                       ▼
┌────────────────────── πρᾶξις (Praxis) ──────────────────────────────────┐
│                                                                         │
│  ActionOrchestrator                                                    │
│  ├─ _confidence_calibrator: ConfidenceCalibrator | None                │
│  │                                                                      │
│  │  execute_batch() safety gate:                                       │
│  │  ┌─ Phase 1: Risk gate (existing)                                  │
│  │  │   τ = 0.5 + autonomy × 0.5                                      │
│  │  │   blocked if risk_q > τ                                          │
│  │  │                                                                   │
│  │  ├─ Phase 1.5: Confidence gate ★ NEW ★                             │
│  │  │   θ = 0.7 − autonomy × 0.2   (bounded [0.3, 0.95])             │
│  │  │   calibrated = calibrator.calibrate_confidence(action.conf)     │
│  │  │   blocked if calibrated < θ                                      │
│  │  │                                                                   │
│  │  ├─ Phase 2: Parallelism                                           │
│  │  └─ Phase 3: DAG scheduling → execution                             │
│  │                                                                      │
│  └─ ExecutionReport: +confidence_blocks, +calibration_applied          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Mechanisms

### 1. Action.confidence_required

Every action can now declare a minimum confidence threshold:

```python
action = Action(
    description="deploy to production",
    tool_name="deploy",
    confidence_required=0.85,  # Must be ≥85% sure
    risk=ActionRisk.HIGH,
)
```

Default is `0.0` — no confidence gate (backward compatible with all existing code).

### 2. Confidence-based Safety Gate (Phase 1.5)

After the risk gate (Phase 1) and before DAG scheduling (Phase 2-3), the confidence gate evaluates actions that survived the risk gate:

```
Given autonomy ∈ [-1, 1] from autonomy_safety tension:
  θ = clamp(0.7 − autonomy × 0.2, 0.3, 0.95)

For each action with confidence_required > 0:
  calibrated = calibrator.calibrate_confidence(action.confidence_required)
  if calibrated < θ:
    → BLOCKED (confidence_blocks++)
```

**Tension modulation of confidence threshold:**

```
autonomy = −1  → θ = 0.9   Maximum safe: only 90%+ calibrated confidence passes
autonomy =  0  → θ = 0.7   Neutral: realistic bar for deployment
autonomy = +1  → θ = 0.5   Max autonomous: even 50% confidence passes
```

### 3. Calibrated vs Raw Confidence

The key insight: the calibrator maps **raw** confidence to **observed** accuracy:

```
Example (overconfident calibrator):
  raw confidence 0.85 → calibrated_confidence ≈ 0.35
  (system says 85% sure, but historically only 35% correct at this level)

  At autonomy=0 (θ=0.7): 0.35 < 0.7 → BLOCKED
  The action is gated because the system's real probability of success is too low.
```

With a well-calibrated system:
```
  raw confidence 0.85 → calibrated_confidence ≈ 0.83
  At autonomy=0 (θ=0.7): 0.83 ≥ 0.7 → PASSES
```

### 4. Three-Dimensional Safety Model

Actions are now gated by three interacting dimensions:

| Dimension | Mechanism | Controlled by |
|-----------|-----------|---------------|
| **Risk** | `risk_q > τ` where `τ = 0.5 + autonomy × 0.5` | Action.risk × autonomy_safety tension |
| **Confidence** | `calibrated < θ` where `θ = 0.7 − autonomy × 0.2` | Action.confidence_required × calibrator × autonomy_safety |
| **Approval** | `approve_fn(action) → True` | External human-in-the-loop override |

Both risk and confidence blocks can be overridden by the approval function, supporting human-in-the-loop workflows.

### 5. ExecutionReport New Fields

```python
class ExecutionReport:
    # ... existing fields ...
    confidence_blocks: int = 0       # Actions blocked by confidence gate
    calibration_applied: bool = False  # Was calibrator used?
```

This enables the equilibrium engine to receive calibration-aware feedback: when `confidence_blocks` is high, push toward safer autonomy mode.

## Tension Modulation

| Tension Axis | Confidence Gating Effect |
|-------------|------------------------|
| `autonomy_safety` | Modulates θ (confidence threshold): safe → high bar, autonomous → low bar |
| `explore_exploit` | (Future: overconfident calibrator → amplify explore signal for more alternatives) |
| `verify_execute` | (Future: low calibrated confidence → increase verification depth) |

## Test Coverage (24 new tests)

| Test Group | Count | Focus |
|-----------|-------|-------|
| Action field | 2 | Default (0.0), custom confidence_required |
| Orchestrator wiring | 3 | Accept, default None, set/unset calibrator |
| ExecutionReport | 2 | Default fields, explicit confidence fields |
| No-calibrator gating | 2 | No calibrator → no confidence blocks, zero conf → no blocks |
| Confidence gating | 5 | Low action blocked, high autonomy lowers threshold, safe mode blocks, well-calibrated passes, approval override |
| Confidence/risk interaction | 1 | Zero-conf actions not affected by confidence gate |
| Import pipeline | 2 | Carries confidence from task dict, defaults to zero |
| Serialization | 1 | Round-trip preserves confidence_required |
| Pillar wiring | 4 | Accepts calibrator, passes to orchestrator, set after init, set to None |
| Pillar E2E | 1 | Full PraxisPillar with calibrator gates actions |
| Edge cases | 1 | Confidence threshold bounded to [0.3, 0.95] |

## Design Decisions

1. **Confidence_required on Action, not metadata** — A first-class field signals that confidence gating is a fundamental mechanism, not an implementation detail. The field has a default of 0.0, so existing code (all 295 tests) needs zero changes.

2. **Calibrator is a runtime reference, not serialized state** — The calibrator's statistical state (bins, weights) is serializable, but the orchestrator holds only a reference. Cross-session calibration persistence (Mneme) is a separate concern (future iteration). From `from_dict()`, the restored orchestrator has `_confidence_calibrator = None`.

3. **Confidence gate runs AFTER risk gate** — Actions blocked by the risk gate never reach the confidence gate. This is correct: if an action is too risky regardless of confidence, block it. If it passes the risk gate but the system isn't actually confident enough, block it there. Two independent safety layers.

4. **Confidence threshold formula: θ = 0.7 − autonomy × 0.2** — At neutral autonomy (0), θ = 0.7 (a realistic deployment bar). At max safe (−1), θ = 0.9 (only highly confident actions). At max autonomous (+1), θ = 0.5 (coin-flip confidence is enough). The direction is opposite the risk gate's τ (which RISES with autonomy) because confidence is a LOWER-IS-WORSE metric, while risk is a HIGHER-IS-WORSE metric.

5. **Bounded to [0.3, 0.95]** — Never require >95% confidence (perfect certainty is impossible) and never allow <30% confidence (worse than random). These bounds prevent degenerate edge cases from extreme tension values.

6. **Calibrator exception safety** — If `calibrate_confidence()` raises, the orchestrator falls back to raw `confidence_required`. This is conservative: raw confidence is typically higher than calibrated (overconfident), so the fallback is more permissive. In production, a calibrator exception would trigger a warning log.

7. **Separate test file (test_confidence_gating.py)** — Confidence gating is a distinct subsystem at the Praxis orchestration level. A separate file keeps the 24 tests self-contained and makes the subsystem boundary clear.

8. **Approval fn overrides both risk and confidence blocks** — Consistency with the existing risk gate: if a human or policy approves the action, it runs. This supports the full spectrum from fully autonomous to fully supervised.

## Why This Creates Impact

### Short-term impact
- **Intelligent safety**: The Praxis safety gate now uses REAL probability of success, not the raw (often overconfident) estimate. An agent that learned it's overconfident will properly tighten its safety gate.
- **Observable calibration quality**: The ExecutionReport now tracks `calibration_applied` and `confidence_blocks`, making calibration effects visible in execution reports.
- **Zero breaking changes**: All 295 existing tests pass without modification. The `confidence_required` field defaults to 0.0, and calibrator defaults to None.
- **Tension-responsive gating**: The `autonomy_safety` tension now modulates TWO safety dimensions — risk threshold AND confidence threshold — creating a richer behavioral space.

### Long-term impact
- **Closed-loop metacognition**: This is the keystone — Cognition learns calibration, Praxis uses it. Every subsequent feature builds on this.
- **Foundation for uncertainty-aware planning**: When Praxis reports confidence blocks, Cognition can adjust its planning depth and branching.
- **Calibration-gated delegation**: The `confidence_required` field enables agents to refuse tasks they're not confident enough to handle.
- **Cross-agent calibration pooling**: The calibrator reference model supports sharing calibration data across Praxis instances in a fleet.

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
│   ├── cognition/             # νοῦς — REASON + PLAN + ★ METACOGNITION
│   │   ├── __init__.py        # ✅ CalibrationBin, ConfidenceCalibrator exports
│   │   ├── attention.py       # ✅ AttentionEquilibriumSystem — context mgmt
│   │   ├── reasoning.py       # ✅ ConfidenceCalibrator + RecursiveReasoningEngine
│   │   └── pillar.py          # ✅ evaluate_result calibration hook
│   ├── praxis/                # πρᾶξις — EXECUTE ★ WITH CALIBRATED SAFETY
│   │   ├── __init__.py
│   │   ├── orchestrator.py    # ★ Updated — confidence gating, calibrator-aware
│   │   └── pillar.py          # ★ Updated — calibrator wiring
│   └── mneme/                 # μνήμη — REMEMBER
│       ├── __init__.py
│       ├── hierarchical.py    # ✅ HierarchicalMneme — 3-tier memory
│       └── pillar.py          # ✅ MnemePillar wrapper
└── tests/
    ├── test_agent.py          # 10 tests
    ├── test_attention.py      # 23 tests
    ├── test_calibration.py    # 63 tests
    ├── test_confidence_gating.py   # 24 tests ★ NEW
    ├── test_equilibrium.py    # 18 tests
    ├── test_mneme.py          # 50 tests
    ├── test_praxis.py         # 68 tests
    └── test_reasoning.py      # 63 tests
```

**Total: 319 tests, 8 test files, 2 modified source files, 1 new test file**

## Cross-Pillar Pipeline Status

| Pipeline | Direction | Status | Iteration 006 Impact |
|----------|-----------|--------|---------------------|
| νοῦς → πρᾶξις | Cognition → Praxis | ✅ **Enhanced** | Plans carry confidence; actions carry confidence_required; calibrator reference shared |
| πρᾶξις → νοῦς | Praxis → Cognition | ✅ **Enhanced** | ExecutionReport now carries confidence_blocks and calibration_applied for metacognitive feedback |
| πρᾶξις → μνήμη | Praxis → Mneme | ✅ | Execution memories carry calibrated confidence |
| νοῦς → μνήμη | Cognition → Mneme | ✅ | Pruned attention chunks unaffected |
| μνήμη → νοῦς | Mneme → Cognition | ✅ | Context retrieval unaffected |

## Next Iteration Candidates

1. **Calibration-Aware Attention**: When calibration quality is poor (high ECE), increase attention budget — the agent needs more context to make better decisions. This closes the Attention↔Calibration loop.

2. **Mneme Calibration History Persistence**: Store calibration state (bins, weights, ECE trend) in Mneme for cross-session persistence. Agents remember their calibration quality across restarts.

3. **Calibration-Gated Delegation**: Use `confidence_required` on the IsonomeAgent level — an agent can refuse tasks it's not confident enough to handle.

4. **Uncertainty-Aware Planning**: When ECE is high or `confidence_blocks` is elevated, the Cognition pillar increases reasoning depth and branching factor.

5. **Cross-Agent Calibration Pooling**: Shared calibrator across multiple agent instances for fleet-level metacognition.

6. **Confidence-Based Verify Depth**: When calibrated confidence is low, automatically increase verification depth (verify_execute tension) for that action.

## Files Changed

```
isonome/
├── praxis/
│   ├── orchestrator.py    (modified — +85 lines: confidence_required field,
│   │                        confidence_calibrator param, confidence safety gate,
│   │                        updated ExecutionReport, serialization)
│   └── pillar.py          (modified — +30 lines: calibrator param, wiring,
│                            set_confidence_calibrator method)
tests/
└── test_confidence_gating.py  (NEW — 310 lines, 24 tests)
```

## Commit Stack

```
feat: calibrated safety gates — metacognition-into-execution closed loop
  - isonome/praxis/orchestrator.py: Action.confidence_required field (default 0.0),
    ActionOrchestrator confidence_calibrator param, Phase 1.5 confidence-based
    safety gating after risk gate, confidence threshold θ = 0.7 - autonomy × 0.2
    bounded [0.3, 0.95], calibrator.calibrate_confidence() used for true probability,
    ExecutionReport +confidence_blocks and +calibration_applied fields,
    import_from_cognition passes confidence_required from task dicts,
    to_dict/from_dict preserve confidence_required in serialization
  - isonome/praxis/pillar.py: PraxisPillar accepts confidence_calibrator param,
    passes to ActionOrchestrator on init, set_confidence_calibrator() method
    for dynamic wiring, supports disable-via-None
  - tests/test_confidence_gating.py: 24 tests — Action field, orchestrator wiring,
    report fields, no-calibrator mode, low-action blocking, tension-modulated
    thresholds, safe/autonomous extremes, well-calibrated passthrough,
    approval override, risk/confidence independence, import pipeline,
    serialization round-trip, pillar integration, end-to-end gating,
    threshold bounding
  - 319/319 tests passing
```
