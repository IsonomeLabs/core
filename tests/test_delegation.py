"""Tests for Calibration-Gated Delegation (iter-019).

Covers:
 - DelegationMode: all five modes, computation logic
 - DelegationDecision: EXECUTE, DELEGATE, SKIP
 - DelegationRecord: frozen dataclass, fields
 - DelegationGate: mode computation, check logic per mode,
   batch checking, statistics, serialization round-trip,
   calibrator replacement, threshold setters
 - Integration: DelegationGate + ActionOrchestrator execute_batch
   Phase 1.7, delegation across tension profiles, well-calibrated
   pass-through, overconfident/underconfident delegation
 - Integration: DelegationGate + PraxisPillar wiring
"""

import pytest
from uuid import UUID, uuid4

from isonome.praxis.delegation import (
    DelegationDecision,
    DelegationGate,
    DelegationMode,
    DelegationOutcome,
    DelegationRecord,
)
from isonome.praxis.orchestrator import (
    Action,
    ActionOrchestrator,
    ActionRisk,
    ActionState,
    ExecutionReport,
)
from isonome.praxis.pillar import PraxisPillar
from isonome.types import AgentIdentity, AgentState, Feedback, Pillar, Signal


# ═══════════════════════════════════════════════════════════════════
# Mock Calibrators
# ═══════════════════════════════════════════════════════════════════


class MockCalibrator:
    """Configurable mock calibrator for testing delegation modes."""

    def __init__(self, ece=0.0, bias=0.0, is_overconfident=False, total_predictions=15):
        self._ece = ece
        self._bias = bias
        self._is_overconfident = is_overconfident
        self.total_predictions = total_predictions

    def compute_ece(self):
        return self._ece

    @property
    def is_overconfident(self):
        return self._is_overconfident

    @property
    def bias(self):
        return self._bias

    def calibrate_confidence(self, c):
        return c * 0.7


class OverconfidentCalibrator(MockCalibrator):
    """ECE=0.25, bias=0.1, overconfident — triggers OVERCONFIDENT mode."""

    def __init__(self):
        super().__init__(ece=0.25, bias=0.1, is_overconfident=True)


class UnderconfidentCalibrator(MockCalibrator):
    """ECE=0.25, bias=-0.15, underconfident — triggers UNDERCONFIDENT mode."""

    def __init__(self):
        super().__init__(ece=0.25, bias=-0.15, is_overconfident=False)


class WellCalibratedCalibrator(MockCalibrator):
    """ECE=0.03, bias=0.01 — triggers WELL_CALIBRATED mode."""

    def __init__(self):
        super().__init__(ece=0.03, bias=0.01, is_overconfident=False)


class ModerateCalibrator(MockCalibrator):
    """ECE=0.10, bias=0.03 — triggers MODERATE mode."""

    def __init__(self):
        super().__init__(ece=0.10, bias=0.03, is_overconfident=False)


class SparseCalibrator(MockCalibrator):
    """Only 3 predictions — triggers UNCALIBRATED mode."""

    def __init__(self):
        super().__init__(ece=0.30, bias=0.2, is_overconfident=True, total_predictions=3)


# ═══════════════════════════════════════════════════════════════════
# DelegationMode tests
# ═══════════════════════════════════════════════════════════════════


class TestDelegationModeComputation:
    """Test compute_mode() under various calibration states."""

    def test_no_calibrator_returns_uncalibrated(self):
        gate = DelegationGate()
        assert gate.compute_mode() == DelegationMode.UNCALIBRATED

    def test_sparse_predictions_returns_uncalibrated(self):
        gate = DelegationGate(calibrator=SparseCalibrator())
        assert gate.compute_mode() == DelegationMode.UNCALIBRATED

    def test_low_ece_returns_well_calibrated(self):
        gate = DelegationGate(calibrator=WellCalibratedCalibrator())
        assert gate.compute_mode() == DelegationMode.WELL_CALIBRATED

    def test_moderate_ece_returns_moderate(self):
        gate = DelegationGate(calibrator=ModerateCalibrator())
        assert gate.compute_mode() == DelegationMode.MODERATE

    def test_high_ece_overconfident(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        assert gate.compute_mode() == DelegationMode.OVERCONFIDENT

    def test_high_ece_underconfident(self):
        gate = DelegationGate(calibrator=UnderconfidentCalibrator())
        assert gate.compute_mode() == DelegationMode.UNDERCONFIDENT

    def test_custom_ece_threshold(self):
        """With ece_threshold=0.05, ModerateCalibrator (ECE=0.10) exceeds threshold.
        Since ModerateCalibrator.is_overconfident=False, mode is UNDERCONFIDENT."""
        gate = DelegationGate(calibrator=ModerateCalibrator(), ece_threshold=0.05)
        assert gate.compute_mode() == DelegationMode.UNDERCONFIDENT

    def test_custom_min_predictions(self):
        """With min_predictions=2, SparseCalibrator (3 preds) becomes OVERCONFIDENT."""
        gate = DelegationGate(calibrator=SparseCalibrator(), min_predictions=2)
        assert gate.compute_mode() == DelegationMode.OVERCONFIDENT

    def test_boundary_ece_005(self):
        """ECE exactly 0.05 should be WELL_CALIBRATED (≤ 0.05)."""
        cal = MockCalibrator(ece=0.05, bias=0.0, is_overconfident=False)
        gate = DelegationGate(calibrator=cal)
        assert gate.compute_mode() == DelegationMode.WELL_CALIBRATED

    def test_boundary_ece_threshold(self):
        """ECE exactly at threshold should be MODERATE (≤ threshold, > 0.05)."""
        cal = MockCalibrator(ece=0.15, bias=0.0, is_overconfident=False)
        gate = DelegationGate(calibrator=cal, ece_threshold=0.15)
        assert gate.compute_mode() == DelegationMode.MODERATE

    def test_just_above_threshold(self):
        """ECE just above threshold should be OVERCONFIDENT/UNDERCONFIDENT."""
        cal = MockCalibrator(ece=0.1501, bias=0.01, is_overconfident=True)
        gate = DelegationGate(calibrator=cal, ece_threshold=0.15)
        assert gate.compute_mode() == DelegationMode.OVERCONFIDENT


# ═══════════════════════════════════════════════════════════════════
# DelegationGate.check() tests
# ═══════════════════════════════════════════════════════════════════


class TestDelegationGateCheck:
    """Test check() decision logic per mode and risk level."""

    def _make_action(self, risk: ActionRisk, description: str = "test"):
        return Action(description=description, tool_name="test_tool", risk=risk)

    # --- UNCALIBRATED: always EXECUTE ---

    def test_uncalibrated_trivial_execute(self):
        gate = DelegationGate(calibrator=SparseCalibrator())
        action = self._make_action(ActionRisk.TRIVIAL)
        assert gate.check(action) == DelegationDecision.EXECUTE

    def test_uncalibrated_critical_execute(self):
        gate = DelegationGate(calibrator=SparseCalibrator())
        action = self._make_action(ActionRisk.CRITICAL)
        assert gate.check(action) == DelegationDecision.EXECUTE

    # --- WELL_CALIBRATED: always EXECUTE ---

    def test_well_calibrated_high_execute(self):
        gate = DelegationGate(calibrator=WellCalibratedCalibrator())
        action = self._make_action(ActionRisk.HIGH)
        assert gate.check(action) == DelegationDecision.EXECUTE

    def test_well_calibrated_critical_execute(self):
        gate = DelegationGate(calibrator=WellCalibratedCalibrator())
        action = self._make_action(ActionRisk.CRITICAL)
        assert gate.check(action) == DelegationDecision.EXECUTE

    # --- MODERATE: always EXECUTE ---

    def test_moderate_high_execute(self):
        gate = DelegationGate(calibrator=ModerateCalibrator())
        action = self._make_action(ActionRisk.HIGH)
        assert gate.check(action) == DelegationDecision.EXECUTE

    # --- OVERCONFIDENT: delegate MODERATE+ ---

    def test_overconfident_trivial_execute(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        action = self._make_action(ActionRisk.TRIVIAL)
        assert gate.check(action) == DelegationDecision.EXECUTE

    def test_overconfident_low_execute(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        action = self._make_action(ActionRisk.LOW)
        assert gate.check(action) == DelegationDecision.EXECUTE

    def test_overconfident_moderate_delegate(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        action = self._make_action(ActionRisk.MODERATE)
        assert gate.check(action) == DelegationDecision.DELEGATE

    def test_overconfident_high_delegate(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        action = self._make_action(ActionRisk.HIGH)
        assert gate.check(action) == DelegationDecision.DELEGATE

    def test_overconfident_critical_delegate(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        action = self._make_action(ActionRisk.CRITICAL)
        assert gate.check(action) == DelegationDecision.DELEGATE

    # --- UNDERCONFIDENT: delegate HIGH+ ---

    def test_underconfident_trivial_execute(self):
        gate = DelegationGate(calibrator=UnderconfidentCalibrator())
        action = self._make_action(ActionRisk.TRIVIAL)
        assert gate.check(action) == DelegationDecision.EXECUTE

    def test_underconfident_moderate_execute(self):
        gate = DelegationGate(calibrator=UnderconfidentCalibrator())
        action = self._make_action(ActionRisk.MODERATE)
        assert gate.check(action) == DelegationDecision.EXECUTE

    def test_underconfident_high_delegate(self):
        gate = DelegationGate(calibrator=UnderconfidentCalibrator())
        action = self._make_action(ActionRisk.HIGH)
        assert gate.check(action) == DelegationDecision.DELEGATE

    def test_underconfident_critical_delegate(self):
        gate = DelegationGate(calibrator=UnderconfidentCalibrator())
        action = self._make_action(ActionRisk.CRITICAL)
        assert gate.check(action) == DelegationDecision.DELEGATE

    # --- Custom thresholds ---

    def test_custom_overconfident_threshold_high_only(self):
        """overconfident_risk_threshold=3 → only HIGH and CRITICAL delegated."""
        gate = DelegationGate(
            calibrator=OverconfidentCalibrator(),
            overconfident_risk_threshold=3,
        )
        action_mod = self._make_action(ActionRisk.MODERATE)
        action_high = self._make_action(ActionRisk.HIGH)
        assert gate.check(action_mod) == DelegationDecision.EXECUTE
        assert gate.check(action_high) == DelegationDecision.DELEGATE

    def test_custom_underconfident_threshold_critical_only(self):
        """underconfident_risk_threshold=4 → only CRITICAL delegated."""
        gate = DelegationGate(
            calibrator=UnderconfidentCalibrator(),
            underconfident_risk_threshold=4,
        )
        action_high = self._make_action(ActionRisk.HIGH)
        action_crit = self._make_action(ActionRisk.CRITICAL)
        assert gate.check(action_high) == DelegationDecision.EXECUTE
        assert gate.check(action_crit) == DelegationDecision.DELEGATE

    # --- risk_value override ---

    def test_risk_value_override(self):
        """Passing risk_value=3 directly should delegate for overconfident."""
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        action = self._make_action(ActionRisk.TRIVIAL)
        # Even though action is TRIVIAL, override risk=3 (HIGH) should delegate
        assert gate.check(action, risk_value=3) == DelegationDecision.DELEGATE

    # --- TRIVIAL always EXECUTE even with high override ---

    # (risk_value override replaces the action's risk, so risk_value=3 with
    #  a TRIVIAL action DOES delegate — this is correct since the override
    #  is intentional. The default path with risk_value=None uses the action's
    #  own risk, and TRIVIAL (0) is always EXECUTE.)


# ═══════════════════════════════════════════════════════════════════
# DelegationRecord tests
# ═══════════════════════════════════════════════════════════════════


class TestDelegationRecord:
    """Test DelegationRecord creation and immutability."""

    def test_record_creation(self):
        record = DelegationRecord(
            action_id=uuid4(),
            action_description="test action",
            action_risk=3,
            mode=DelegationMode.OVERCONFIDENT,
            decision=DelegationDecision.DELEGATE,
            ece=0.25,
            bias=0.1,
            reason="overconfident + risk_value=3 ≥ 2 — delegate",
        )
        assert record.action_risk == 3
        assert record.decision == DelegationDecision.DELEGATE
        assert record.mode == DelegationMode.OVERCONFIDENT

    def test_record_is_frozen(self):
        record = DelegationRecord(
            action_id=uuid4(),
            action_description="test",
            action_risk=0,
            mode=DelegationMode.WELL_CALIBRATED,
            decision=DelegationDecision.EXECUTE,
            ece=0.0,
            bias=0.0,
            reason="test",
        )
        with pytest.raises(AttributeError):
            record.action_risk = 5

    def test_record_has_slots(self):
        record = DelegationRecord(
            action_id=uuid4(),
            action_description="test",
            action_risk=0,
            mode=DelegationMode.UNCALIBRATED,
            decision=DelegationDecision.EXECUTE,
            ece=0.0,
            bias=0.0,
            reason="test",
        )
        assert hasattr(record, "__slots__")


# ═══════════════════════════════════════════════════════════════════
# DelegationGate statistics tests
# ═══════════════════════════════════════════════════════════════════


class TestDelegationGateStats:
    """Test stats tracking and delegation_rate."""

    def test_initial_stats(self):
        gate = DelegationGate()
        stats = gate.stats
        assert stats["total_checks"] == 0
        assert stats["total_delegated"] == 0
        assert stats["total_executed"] == 0
        assert stats["delegation_rate"] == 0.0

    def test_stats_after_checks(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        for risk in [ActionRisk.TRIVIAL, ActionRisk.LOW, ActionRisk.MODERATE, ActionRisk.HIGH]:
            action = Action(description=f"t-{risk.name}", tool_name="t", risk=risk)
            gate.check(action)

        stats = gate.stats
        assert stats["total_checks"] == 4
        assert stats["total_delegated"] == 2  # MODERATE + HIGH
        assert stats["total_executed"] == 2  # TRIVIAL + LOW
        assert stats["delegation_rate"] == 0.5

    def test_delegation_rate_property(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        action = Action(description="t", tool_name="t", risk=ActionRisk.HIGH)
        gate.check(action)
        assert gate.delegation_rate == 1.0

    def test_records_property_immutable(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        action = Action(description="t", tool_name="t", risk=ActionRisk.HIGH)
        gate.check(action)
        records = gate.records
        assert isinstance(records, tuple)
        assert len(records) == 1
        # Original list should not be affected by tuple creation
        assert len(gate.records) == 1

    def test_stats_include_ece_and_bias(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        stats = gate.stats
        assert stats["ece"] == 0.25
        assert stats["bias"] == 0.1

    def test_stats_no_calibrator(self):
        gate = DelegationGate()
        stats = gate.stats
        assert stats["ece"] == 0.0
        assert stats["bias"] == 0.0
        assert stats["calibrator_predictions"] == 0


# ═══════════════════════════════════════════════════════════════════
# DelegationGate batch checking
# ═══════════════════════════════════════════════════════════════════


class TestDelegationGateBatch:
    """Test check_batch() with multiple actions."""

    def test_batch_returns_dict(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        actions = [
            Action(description="t1", tool_name="t", risk=ActionRisk.TRIVIAL),
            Action(description="t2", tool_name="t", risk=ActionRisk.HIGH),
        ]
        results = gate.check_batch(actions)
        assert len(results) == 2
        assert all(isinstance(k, UUID) for k in results.keys())

    def test_batch_correct_decisions(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        actions = [
            Action(description="trivial", tool_name="t", risk=ActionRisk.TRIVIAL),
            Action(description="low", tool_name="t", risk=ActionRisk.LOW),
            Action(description="moderate", tool_name="t", risk=ActionRisk.MODERATE),
            Action(description="high", tool_name="t", risk=ActionRisk.HIGH),
            Action(description="critical", tool_name="t", risk=ActionRisk.CRITICAL),
        ]
        results = gate.check_batch(actions)
        decisions = [results[a.id] for a in actions]
        assert decisions[0] == DelegationDecision.EXECUTE  # TRIVIAL
        assert decisions[1] == DelegationDecision.EXECUTE  # LOW
        assert decisions[2] == DelegationDecision.DELEGATE  # MODERATE
        assert decisions[3] == DelegationDecision.DELEGATE  # HIGH
        assert decisions[4] == DelegationDecision.DELEGATE  # CRITICAL

    def test_batch_empty(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        results = gate.check_batch([])
        assert results == {}


# ═══════════════════════════════════════════════════════════════════
# DelegationGate serialization
# ═══════════════════════════════════════════════════════════════════


class TestDelegationGateSerialization:
    """Test to_dict() and from_dict() round-trip."""

    def test_round_trip_basic(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        # Run some checks to populate stats
        action = Action(description="test", tool_name="t", risk=ActionRisk.HIGH)
        gate.check(action)

        data = gate.to_dict()
        gate2 = DelegationGate.from_dict(data, calibrator=OverconfidentCalibrator())

        assert gate2._ece_threshold == gate._ece_threshold
        assert gate2._min_predictions == gate._min_predictions
        assert gate2._overconfident_risk_threshold == gate._overconfident_risk_threshold
        assert gate2._underconfident_risk_threshold == gate._underconfident_risk_threshold
        assert gate2._total_checks == gate._total_checks
        assert gate2._total_delegated == gate._total_delegated

    def test_round_trip_preserves_records(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        actions = [
            Action(description="trivial", tool_name="t", risk=ActionRisk.TRIVIAL),
            Action(description="high", tool_name="t", risk=ActionRisk.HIGH),
        ]
        for a in actions:
            gate.check(a)

        data = gate.to_dict()
        gate2 = DelegationGate.from_dict(data, calibrator=OverconfidentCalibrator())

        assert len(gate2._records) == 2
        assert gate2._records[0].action_description == "trivial"
        assert gate2._records[0].decision == DelegationDecision.EXECUTE
        assert gate2._records[1].action_description == "high"
        assert gate2._records[1].decision == DelegationDecision.DELEGATE

    def test_round_trip_custom_thresholds(self):
        gate = DelegationGate(
            calibrator=OverconfidentCalibrator(),
            ece_threshold=0.10,
            min_predictions=5,
            overconfident_risk_threshold=3,
            underconfident_risk_threshold=4,
        )
        data = gate.to_dict()
        gate2 = DelegationGate.from_dict(data, calibrator=OverconfidentCalibrator())
        assert gate2._ece_threshold == 0.10
        assert gate2._min_predictions == 5
        assert gate2._overconfident_risk_threshold == 3
        assert gate2._underconfident_risk_threshold == 4

    def test_round_trip_without_calibrator(self):
        gate = DelegationGate()
        data = gate.to_dict()
        gate2 = DelegationGate.from_dict(data)
        assert gate2._calibrator is None
        assert gate2.compute_mode() == DelegationMode.UNCALIBRATED

    def test_round_trip_handles_malformed_records(self):
        """Malformed records in data should be skipped gracefully."""
        data = {
            "ece_threshold": 0.15,
            "records": [
                {"action_id": "not-a-uuid", "mode": "OVERCONFIDENT", "decision": "DELEGATE"},
            ],
        }
        gate = DelegationGate.from_dict(data)
        assert len(gate._records) == 0  # Skipped the bad record

    def test_to_dict_includes_all_config(self):
        gate = DelegationGate(ece_threshold=0.20, min_predictions=20)
        data = gate.to_dict()
        assert "ece_threshold" in data
        assert "min_predictions" in data
        assert "overconfident_risk_threshold" in data
        assert "underconfident_risk_threshold" in data
        assert "records" in data

    def test_from_dict_defaults(self):
        """Missing keys should use defaults."""
        gate = DelegationGate.from_dict({})
        assert gate._ece_threshold == 0.15
        assert gate._min_predictions == 10


# ═══════════════════════════════════════════════════════════════════
# DelegationGate configuration
# ═══════════════════════════════════════════════════════════════════


class TestDelegationGateConfig:
    """Test configuration setters and calibrator management."""

    def test_set_calibrator(self):
        gate = DelegationGate()
        assert gate.compute_mode() == DelegationMode.UNCALIBRATED
        gate.set_calibrator(OverconfidentCalibrator())
        assert gate.compute_mode() == DelegationMode.OVERCONFIDENT

    def test_replace_calibrator(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        assert gate.compute_mode() == DelegationMode.OVERCONFIDENT
        gate.set_calibrator(WellCalibratedCalibrator())
        assert gate.compute_mode() == DelegationMode.WELL_CALIBRATED

    def test_ece_threshold_setter_clamps(self):
        gate = DelegationGate()
        gate.ece_threshold = -0.5
        assert gate.ece_threshold == 0.0
        gate.ece_threshold = 2.0
        assert gate.ece_threshold == 1.0

    def test_ece_threshold_setter_valid(self):
        gate = DelegationGate()
        gate.ece_threshold = 0.20
        assert gate.ece_threshold == 0.20

    def test_min_predictions_setter_clamps(self):
        gate = DelegationGate()
        gate.min_predictions = -5
        assert gate.min_predictions == 1

    def test_min_predictions_setter_valid(self):
        gate = DelegationGate()
        gate.min_predictions = 50
        assert gate.min_predictions == 50

    def test_repr(self):
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        r = repr(gate)
        assert "OVERCONFIDENT" in r
        assert "DelegationGate" in r


# ═══════════════════════════════════════════════════════════════════
# Orchestrator integration: Phase 1.7
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorDelegationIntegration:
    """Test DelegationGate integration with ActionOrchestrator.execute_batch."""

    def _make_permissive_orchestrator(self, calibrator):
        """Create orchestrator with permissive tension profile."""
        orch = ActionOrchestrator()
        gate = DelegationGate(calibrator=calibrator)
        orch.set_delegation_gate(gate=gate)
        orch.set_tension_profile({
            "autonomy_safety": 0.5,
            "sequential_parallel": 0.0,
            "verify_execute": 0.0,
        })
        return orch

    def test_overconfident_delegates_moderate_and_high(self):
        orch = self._make_permissive_orchestrator(OverconfidentCalibrator())
        for risk in [ActionRisk.TRIVIAL, ActionRisk.LOW, ActionRisk.MODERATE, ActionRisk.HIGH]:
            orch.register_action(Action(description=f"t-{risk.name}", tool_name="t", risk=risk))
        report = orch.execute_batch(lambda a: "ok")
        assert report.delegation_active is True
        assert report.delegation_count == 2  # MODERATE + HIGH
        assert report.delegation_mode == "OVERCONFIDENT"

    def test_well_calibrated_no_delegation(self):
        orch = self._make_permissive_orchestrator(WellCalibratedCalibrator())
        # Use very permissive profile so CRITICAL passes risk gate
        orch.set_tension_profile({"autonomy_safety": 1.0, "sequential_parallel": 0.0, "verify_execute": 0.0})
        for risk in [ActionRisk.TRIVIAL, ActionRisk.LOW, ActionRisk.MODERATE, ActionRisk.HIGH, ActionRisk.CRITICAL]:
            orch.register_action(Action(description=f"t-{risk.name}", tool_name="t", risk=risk))
        report = orch.execute_batch(lambda a: "ok")
        assert report.delegation_count == 0
        assert report.actions_completed == 5

    def test_underconfident_delegates_high_and_critical(self):
        orch = self._make_permissive_orchestrator(UnderconfidentCalibrator())
        orch.set_tension_profile({
            "autonomy_safety": 1.0,  # Very permissive
            "sequential_parallel": 0.0,
            "verify_execute": 0.0,
        })
        for risk in [ActionRisk.TRIVIAL, ActionRisk.LOW, ActionRisk.MODERATE, ActionRisk.HIGH, ActionRisk.CRITICAL]:
            orch.register_action(Action(description=f"t-{risk.name}", tool_name="t", risk=risk))
        report = orch.execute_batch(lambda a: "ok")
        assert report.delegation_count == 2  # HIGH + CRITICAL
        assert report.delegation_mode == "UNDERCONFIDENT"

    def test_no_gate_no_delegation(self):
        orch = ActionOrchestrator()
        orch.set_tension_profile({"autonomy_safety": 1.0, "sequential_parallel": 0.0, "verify_execute": 0.0})
        for risk in [ActionRisk.TRIVIAL, ActionRisk.HIGH, ActionRisk.CRITICAL]:
            orch.register_action(Action(description=f"t-{risk.name}", tool_name="t", risk=risk))
        report = orch.execute_batch(lambda a: "ok")
        assert report.delegation_active is False
        assert report.delegation_count == 0
        assert report.actions_completed == 3

    def test_uncalibrated_no_delegation(self):
        orch = self._make_permissive_orchestrator(SparseCalibrator())
        for risk in [ActionRisk.TRIVIAL, ActionRisk.HIGH]:
            orch.register_action(Action(description=f"t-{risk.name}", tool_name="t", risk=risk))
        report = orch.execute_batch(lambda a: "ok")
        assert report.delegation_count == 0
        assert report.delegation_mode == "UNCALIBRATED"

    def test_delegated_actions_are_blocked(self):
        orch = self._make_permissive_orchestrator(OverconfidentCalibrator())
        orch.register_action(Action(description="high-task", tool_name="t", risk=ActionRisk.HIGH))
        orch.register_action(Action(description="low-task", tool_name="t", risk=ActionRisk.LOW))
        report = orch.execute_batch(lambda a: "ok")
        # HIGH delegated → blocked, LOW executed
        assert report.actions_completed == 1
        assert report.delegation_count == 1

    def test_delegation_and_risk_gate_interact(self):
        """Risk gate blocks CRITICAL; delegation gate delegates MODERATE+ from remaining."""
        orch = self._make_permissive_orchestrator(OverconfidentCalibrator())
        # autonomy_safety=0.5 → tau=0.75; CRITICAL (risk_q=1.0) blocked by risk gate
        # MODERATE (risk_q=0.5) passes risk gate, then delegated by gate
        orch.register_action(Action(description="crit", tool_name="t", risk=ActionRisk.CRITICAL))
        orch.register_action(Action(description="mod", tool_name="t", risk=ActionRisk.MODERATE))
        orch.register_action(Action(description="low", tool_name="t", risk=ActionRisk.LOW))
        report = orch.execute_batch(lambda a: "ok")
        # CRITICAL blocked by risk gate, MODERATE delegated, LOW completed
        assert report.actions_completed == 1
        assert report.delegation_count == 1

    def test_set_delegation_gate_with_calibrator_only(self):
        """set_delegation_gate(calibrator=...) creates a gate automatically."""
        orch = ActionOrchestrator()
        orch.set_delegation_gate(calibrator=OverconfidentCalibrator())
        assert orch._delegation_gate is not None

    def test_set_delegation_gate_none_disables(self):
        orch = ActionOrchestrator()
        orch.set_delegation_gate(gate=DelegationGate(calibrator=OverconfidentCalibrator()))
        assert orch._delegation_gate is not None
        orch.set_delegation_gate(gate=None)
        assert orch._delegation_gate is None


# ═══════════════════════════════════════════════════════════════════
# PraxisPillar wiring
# ═══════════════════════════════════════════════════════════════════


class TestPraxisPillarDelegation:
    """Test that PraxisPillar correctly wires the delegation gate."""

    def test_pillar_creates_gate_with_calibrator(self):
        """PraxisPillar with delegation_gate param should wire it through."""
        cal = OverconfidentCalibrator()
        gate = DelegationGate(calibrator=cal)
        pillar = PraxisPillar(delegation_gate=gate)
        # Initialize the pillar to create the orchestrator
        identity = AgentIdentity(agent_id="test", name="test")
        state = AgentState(identity=identity)
        pillar.initialize(state)
        # The pillar should have the delegation gate wired to its orchestrator
        assert pillar.orchestrator._delegation_gate is not None

    def test_pillar_without_delegation_gate(self):
        pillar = PraxisPillar()
        identity = AgentIdentity(agent_id="test", name="test")
        state = AgentState(identity=identity)
        pillar.initialize(state)
        assert pillar.orchestrator._delegation_gate is None


# ═══════════════════════════════════════════════════════════════════
# Cross-pillar integration: Cognition → Praxis delegation
# ═══════════════════════════════════════════════════════════════════


class TestCrossPillarDelegation:
    """Test that calibration state from Cognition drives delegation in Praxis."""

    def test_calibrator_change_updates_delegation(self):
        """Changing the calibrator should change delegation behavior."""
        gate = DelegationGate(calibrator=WellCalibratedCalibrator())
        action = Action(description="high", tool_name="t", risk=ActionRisk.HIGH)

        # Well-calibrated → EXECUTE
        assert gate.check(action) == DelegationDecision.EXECUTE

        # Switch to overconfident
        gate.set_calibrator(OverconfidentCalibrator())
        action2 = Action(description="high2", tool_name="t", risk=ActionRisk.HIGH)
        assert gate.check(action2) == DelegationDecision.DELEGATE

    def test_delegation_feedback_loop(self):
        """Delegated actions should be recorded for feedback to Cognition."""
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        action = Action(description="deploy", tool_name="deploy", risk=ActionRisk.HIGH)
        gate.check(action)

        records = gate.records
        assert len(records) == 1
        record = records[0]
        assert record.decision == DelegationDecision.DELEGATE
        assert record.ece == 0.25
        assert record.bias == 0.1
        assert "overconfident" in record.reason.lower()

    def test_delegation_stats_for_feedback(self):
        """Stats should be accessible for cross-pillar feedback."""
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        for risk in [ActionRisk.TRIVIAL, ActionRisk.LOW, ActionRisk.MODERATE, ActionRisk.HIGH]:
            action = Action(description=f"t-{risk.name}", tool_name="t", risk=risk)
            gate.check(action)

        stats = gate.stats
        assert stats["delegation_rate"] == 0.5  # 2 of 4 delegated
        assert stats["mode"] == "OVERCONFIDENT"
        assert stats["total_delegated"] == 2


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestDelegationEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_ece(self):
        cal = MockCalibrator(ece=0.0, bias=0.0, is_overconfident=False)
        gate = DelegationGate(calibrator=cal)
        assert gate.compute_mode() == DelegationMode.WELL_CALIBRATED

    def test_max_ece(self):
        cal = MockCalibrator(ece=1.0, bias=0.5, is_overconfident=True, total_predictions=100)
        gate = DelegationGate(calibrator=cal)
        assert gate.compute_mode() == DelegationMode.OVERCONFIDENT
        action = Action(description="t", tool_name="t", risk=ActionRisk.LOW)
        # Even LOW risk (1) is below threshold 2 → EXECUTE
        assert gate.check(action) == DelegationDecision.EXECUTE

    def test_delegated_count_in_blocked_count(self):
        """Delegated actions also count as blocked in the report."""
        orch = ActionOrchestrator()
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        orch.set_delegation_gate(gate=gate)
        orch.set_tension_profile({"autonomy_safety": 1.0, "sequential_parallel": 0.0, "verify_execute": 0.0})
        orch.register_action(Action(description="mod", tool_name="t", risk=ActionRisk.MODERATE))
        orch.register_action(Action(description="low", tool_name="t", risk=ActionRisk.LOW))
        report = orch.execute_batch(lambda a: "ok")
        # MODERATE delegated (counted as blocked too), LOW completed
        assert report.delegation_count == 1
        assert report.actions_blocked == 1
        assert report.actions_completed == 1

    def test_all_actions_delegated(self):
        """All actions above threshold → all delegated, none completed."""
        orch = ActionOrchestrator()
        gate = DelegationGate(calibrator=OverconfidentCalibrator(), overconfident_risk_threshold=0)
        # threshold=0 means even TRIVIAL gets delegated? No — TRIVIAL (0) is always EXECUTE
        # But risk 0 == 0, so 0 >= 0 is true... let's check the implementation
        orch.set_delegation_gate(gate=gate)
        orch.set_tension_profile({"autonomy_safety": 1.0, "sequential_parallel": 0.0, "verify_execute": 0.0})
        orch.register_action(Action(description="trivial", tool_name="t", risk=ActionRisk.TRIVIAL))
        orch.register_action(Action(description="low", tool_name="t", risk=ActionRisk.LOW))
        report = orch.execute_batch(lambda a: "ok")
        # TRIVIAL is hard-coded to always EXECUTE regardless of threshold
        assert report.actions_completed == 1  # TRIVIAL

    def test_moderate_mode_no_delegation_even_high_risk(self):
        """MODERATE mode should never delegate regardless of risk."""
        gate = DelegationGate(calibrator=ModerateCalibrator())
        action = Action(description="critical", tool_name="t", risk=ActionRisk.CRITICAL)
        assert gate.check(action) == DelegationDecision.EXECUTE

    def test_gate_records_accumulate(self):
        """Records accumulate across multiple check calls."""
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        for _ in range(10):
            action = Action(description="t", tool_name="t", risk=ActionRisk.HIGH)
            gate.check(action)
        assert len(gate.records) == 10
        assert gate._total_delegated == 10



# ═══════════════════════════════════════════════════════════════════
# Orchestrator serialization with delegation gate
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorDelegationSerialization:
    """Test that orchestrator to_dict/from_dict preserves delegation gate."""

    def test_to_dict_includes_delegation_gate(self):
        orch = ActionOrchestrator()
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        orch.set_delegation_gate(gate=gate)
        orch.register_action(Action(description="test", tool_name="t", risk=ActionRisk.LOW))
        orch.execute_batch(lambda a: "ok")
        data = orch.to_dict()
        assert "delegation_gate" in data
        assert data["delegation_gate"]["ece_threshold"] == 0.15

    def test_to_dict_no_delegation_gate(self):
        orch = ActionOrchestrator()
        data = orch.to_dict()
        assert "delegation_gate" not in data

    def test_from_dict_restores_gate_config(self):
        orch = ActionOrchestrator()
        gate = DelegationGate(
            calibrator=OverconfidentCalibrator(),
            ece_threshold=0.20,
            overconfident_risk_threshold=3,
        )
        orch.set_delegation_gate(gate=gate)
        action = Action(description="test", tool_name="t", risk=ActionRisk.HIGH)
        orch.register_action(action)
        orch.execute_batch(lambda a: "ok")
        data = orch.to_dict()
        # Restore without calibrator (calibrators aren't serialized)
        orch2 = ActionOrchestrator.from_dict(data)
        assert orch2._delegation_gate is not None
        assert orch2._delegation_gate._ece_threshold == 0.20
        assert orch2._delegation_gate._overconfident_risk_threshold == 3

    def test_from_dict_without_gate_data(self):
        """Orchestrator without delegation gate should deserialize cleanly."""
        orch = ActionOrchestrator()
        orch.register_action(Action(description="test", tool_name="t", risk=ActionRisk.LOW))
        orch.execute_batch(lambda a: "ok")
        data = orch.to_dict()
        orch2 = ActionOrchestrator.from_dict(data)
        assert orch2._delegation_gate is None

    def test_from_dict_restores_gate_stats(self):
        """Gate cumulative stats should survive serialization round-trip."""
        orch = ActionOrchestrator()
        gate = DelegationGate(calibrator=OverconfidentCalibrator())
        orch.set_delegation_gate(gate=gate)
        for risk in [ActionRisk.TRIVIAL, ActionRisk.LOW, ActionRisk.MODERATE, ActionRisk.HIGH]:
            orch.register_action(Action(description=f"t-{risk.name}", tool_name="t", risk=risk))
        orch.set_tension_profile({"autonomy_safety": 1.0, "sequential_parallel": 0.0, "verify_execute": 0.0})
        orch.execute_batch(lambda a: "ok")
        data = orch.to_dict()
        orch2 = ActionOrchestrator.from_dict(data)
        assert orch2._delegation_gate._total_checks == 4
        assert orch2._delegation_gate._total_delegated == 2
