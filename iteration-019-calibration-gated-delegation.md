# Iteration 019: Calibration-Gated Delegation

**Date**: 2026-06-04
**Pillar**: Praxis (cross-pillar: Cognition → Praxis)
**Status**: COMPLETE
**Tests**: 895 passing (74 new delegation tests)

---

## Summary

When the confidence calibrator reports poor calibration (high ECE), the system should not trust its own execution judgments for high-risk actions. This iteration adds a **DelegationGate** to the Praxis pillar that routes high-risk actions to subagents instead of executing them directly when the system is poorly calibrated. This creates a new metacognitive feedback loop: Cognition → Praxis delegation, and Praxis → Cognition delegation records.

## Problem

The existing orchestrator had two safety gating phases:
- **Phase 1 (Risk Gate)**: Blocks actions exceeding the autonomy_safety tension threshold
- **Phase 1.5 (Confidence Gate)**: Blocks actions whose calibrated confidence falls below a threshold

However, neither phase addresses the case where the system is **systematically miscalibrated**. An overconfident system might have high raw confidence in dangerous actions — the confidence gate only checks if calibrated confidence meets a threshold, not whether the calibration system itself is trustworthy. When the calibrator's ECE is high, the system shouldn't trust any of its confidence estimates enough to autonomously execute risky actions.

## Solution: DelegationGate

### Core Mechanism

```
ECE > θ_delegate  →  delegate actions at risk level ≥ R_threshold
ECE ≤ θ_delegate  →  execute all actions normally (gate open)
```

Where:
- `θ_delegate`: ECE threshold for activation (default: 0.15)
- `R_threshold`: minimum risk level for delegation
  - Overconfident: `R_threshold = MODERATE (2)` — the system over-estimates its capability
  - Underconfident: `R_threshold = HIGH (3)` — the system may underestimate, but risk classifications are still trusted

### Five Operating Modes

| Mode | Condition | Behavior |
|------|-----------|----------|
| UNCALIBRATED | < 10 predictions | Execute all (insufficient data) |
| WELL_CALIBRATED | ECE ≤ 0.05 | Execute all (gate open) |
| MODERATE | 0.05 < ECE ≤ 0.15 | Execute all (not severe enough) |
| OVERCONFIDENT | ECE > 0.15, bias > 0 | Delegate MODERATE+ risk |
| UNDERCONFIDENT | ECE > 0.15, bias ≤ 0 | Delegate HIGH+ risk |

### Why Overconfident Delegates at Lower Threshold

An overconfident system (bias > 0) systematically overestimates its own capability. Its confidence estimates for high-risk actions are inflated — it might report 0.9 confidence for an action that actually succeeds only 0.6 of the time. Therefore, we don't trust even MODERATE-risk assessments from an overconfident system, and delegate them.

An underconfident system (bias ≤ 0) is being too cautious. Its risk classifications are likely still valid — it's just overly hesitant. We still trust its judgment about which actions are HIGH vs. MODERATE risk, so we only delegate the clearly dangerous ones.

### Integration Point: Phase 1.7

The delegation gate runs as **Phase 1.7** in `execute_batch()`, between the confidence gate (Phase 1.5) and the parallelism computation (Phase 2):

```
Phase 1   → Risk Gate (autonomy_safety tension)
Phase 1.5 → Confidence Gate (calibrated confidence threshold)
Phase 1.7 → Delegation Gate (calibration quality check)  ← NEW
Phase 2   → Parallelism (sequential_parallel tension)
Phase 2.5 → Verify Depth (verify_execute tension)
Phase 3+  → Execution, retry, validation
```

Delegated actions are removed from the current batch (marked BLOCKED) and tracked separately via DelegationRecord. They count as "blocked" in the ExecutionReport but are distinguished by `delegation_count` and `delegation_mode` fields.

## Cross-Pillar Integration

### Cognition → Praxis
The calibrator's ECE and bias flow from the Cognition pillar through the shared `confidence_calibrator` reference. When Praxis reads the calibrator during `execute_batch()`, it checks whether delegation is warranted.

### Praxis → Cognition
Delegated action decisions generate DelegationRecords with (predicted_confidence, delegated=True) pairs. These feed back to Cognition as calibration signals — the system chose not to execute autonomously, which is information about its own self-assessment accuracy.

### Praxis → Mneme
Delegated action outcomes are stored with a "delegated" tag for future pattern matching and learning.

## Files Changed

| File | Change |
|------|--------|
| `isonome/praxis/delegation.py` | **NEW** (407 lines): DelegationGate, DelegationMode, DelegationDecision, DelegationRecord |
| `isonome/praxis/__init__.py` | Added DelegationGate exports |
| `isonome/praxis/orchestrator.py` | Phase 1.7 delegation gate, ExecutionReport delegation fields, set_delegation_gate(), serialization |
| `isonome/praxis/pillar.py` | Added delegation_gate parameter, wire to orchestrator, set_delegation_gate() method |
| `tests/test_delegation.py` | **NEW** (74 tests): comprehensive coverage of all modes, edge cases, integration |

## Mathematical Foundation

### Delegation Decision Function

```
D(action, calibrator) =
  if |calibrator.predictions| < min_predictions:
    EXECUTE                              # insufficient data
  elif ECE ≤ 0.05:
    EXECUTE                              # well-calibrated
  elif ECE ≤ θ_delegate:
    EXECUTE                              # moderate miscalibration
  elif calibrator.is_overconfident:
    DELEGATE if action.risk ≥ R_overconfident   # default: 2 (MODERATE)
    EXECUTE otherwise
  else:
    DELEGATE if action.risk ≥ R_underconfident  # default: 3 (HIGH)
    EXECUTE otherwise
```

### TRIVIAL Risk Exception
Actions with TRIVIAL risk (value=0) always execute directly regardless of calibration state. There is no meaningful benefit to delegating a no-side-effect action.

### Interaction with Risk Gate
The delegation gate only evaluates actions that passed the risk gate (Phase 1). Actions already blocked by the risk gate are not considered for delegation — there is no point delegating an action that the system's safety profile already blocks.

## Test Coverage

74 new tests across 10 test classes:

1. **TestDelegationModeComputation** (10 tests): All five modes, boundary conditions, custom thresholds
2. **TestDelegationGateCheck** (16 tests): Decision logic per mode × risk level, custom thresholds, risk_value override
3. **TestDelegationRecord** (3 tests): Creation, immutability, slots
4. **TestDelegationGateStats** (6 tests): Statistics tracking, delegation_rate, records immutability
5. **TestDelegationGateBatch** (3 tests): Batch checking, empty batch
6. **TestDelegationGateSerialization** (6 tests): Round-trip, records preservation, custom thresholds, malformed data
7. **TestDelegationGateConfig** (6 tests): Calibrator management, threshold setters, repr
8. **TestOrchestratorDelegationIntegration** (10 tests): Phase 1.7 in execute_batch, tension profiles, gate interaction
9. **TestPraxisPillarDelegation** (2 tests): Pillar wiring, no-gate case
10. **TestCrossPillarDelegation** (3 tests): Calibrator change → delegation change, feedback loop, stats
11. **TestDelegationEdgeCases** (7 tests): Zero/max ECE, all-delegated, MODERATE mode, accumulating records

## Remaining Gaps

From iteration-018 candidate list, these remain:
1. **Cross-agent calibration pooling**: Share calibration data across agent instances
2. **Attention weight rebalancing**: Already done in iter-016 ✓

## Next Iteration Candidates

1. **Cross-agent calibration pooling**: Multi-agent ECE aggregation for faster convergence
2. **Delegation outcome tracking**: When delegated actions complete, feed results back to calibrator
3. **Dynamic threshold adaptation**: Adjust θ_delegate based on system performance over time
