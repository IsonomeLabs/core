"""Tests for Delegation Outcome Tracking (iter-020).

Covers:
- DelegationOutcome: frozen dataclass, fields, defaults
- DelegationGate.record_outcome(): recording, calibrator feedback,
  threshold adaptation
- DelegationGate.delegation_accuracy: rolling window computation
- DelegationGate.reset_threshold(): reset to initial value
- DelegationGate serialization round-trip with outcomes
- Threshold adaptation: tighten/loosen/no-op based on accuracy
- Edge cases: no calibrator, no outcomes, feedback disabled,
  boundary conditions
"""

import pytest
from uuid import uuid4

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
)


# ═══════════════════════════════════════════════════════════════════
# Mock Calibrators with record() support
# ═══════════════════════════════════════════════════════════════════


class RecordingMockCalibrator:
    """Mock calibrator that supports record() for outcome feedback tests."""

    def __init__(self, ece=0.25, bias=0.1, is_overconfident=True, total_predictions=15):
        self._ece = ece
        self._bias = bias
        self._is_overconfident = is_overconfident
        self.total_predictions = total_predictions
        self.recorded: list[tuple[float, bool]] = []

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

    def record(self, predicted_confidence: float, actual_success: bool) -> None:
        self.recorded.append((predicted_confidence, actual_success))
        self.total_predictions += 1


class RecordingOverconfidentCalibrator(RecordingMockCalibrator):
    """ECE=0.25, bias=0.1, overconfident — triggers OVERCONFIDENT mode."""

    def __init__(self):
        super().__init__(ece=0.25, bias=0.1, is_overconfident=True)


class RecordingUnderconfidentCalibrator(RecordingMockCalibrator):
    """ECE=0.25, bias=-0.15, underconfident — triggers UNDERCONFIDENT mode."""

    def __init__(self):
        super().__init__(ece=0.25, bias=-0.15, is_overconfident=False)


class RecordingWellCalibratedCalibrator(RecordingMockCalibrator):
    """ECE=0.03 — triggers WELL_CALIBRATED mode."""

    def __init__(self):
        super().__init__(ece=0.03, bias=0.01, is_overconfident=False)


def _make_outcome(
    *,
    actual_success: bool = True,
    predicted_confidence: float = 0.6,
    feedback_to_calibrator: bool = True,
    action_risk: int = 3,
) -> DelegationOutcome:
    """Create a DelegationOutcome with sensible defaults for testing."""
    return DelegationOutcome(
        action_id=uuid4(),
        action_description="test action",
        action_risk=action_risk,
        predicted_confidence=predicted_confidence,
        actual_success=actual_success,
        delegated_mode=DelegationMode.OVERCONFIDENT,
        ece_at_delegation=0.25,
        feedback_to_calibrator=feedback_to_calibrator,
    )


# ═══════════════════════════════════════════════════════════════════
# DelegationOutcome dataclass tests
# ═══════════════════════════════════════════════════════════════════


class TestDelegationOutcome:
    """Tests for the DelegationOutcome frozen dataclass."""

    def test_outcome_is_frozen(self):
        """DelegationOutcome should be immutable."""
        o = _make_outcome()
        with pytest.raises(AttributeError):
            o.actual_success = False  # type: ignore[misc]

    def test_outcome_fields(self):
        """All fields should be accessible."""
        action_id = uuid4()
        o = DelegationOutcome(
            action_id=action_id,
            action_description="deploy to prod",
            action_risk=4,
            predicted_confidence=0.7,
            actual_success=False,
            delegated_mode=DelegationMode.UNDERCONFIDENT,
            ece_at_delegation=0.18,
            feedback_to_calibrator=False,
        )
        assert o.action_id == action_id
        assert o.action_description == "deploy to prod"
        assert o.action_risk == 4
        assert o.predicted_confidence == 0.7
        assert o.actual_success is False
        assert o.delegated_mode == DelegationMode.UNDERCONFIDENT
        assert o.ece_at_delegation == 0.18
        assert o.feedback_to_calibrator is False

    def test_outcome_default_feedback(self):
        """feedback_to_calibrator should default to True."""
        o = DelegationOutcome(
            action_id=uuid4(),
            action_description="test",
            action_risk=2,
            predicted_confidence=0.5,
            actual_success=True,
            delegated_mode=DelegationMode.OVERCONFIDENT,
            ece_at_delegation=0.2,
        )
        assert o.feedback_to_calibrator is True

    def test_outcome_has_slots(self):
        """DelegationOutcome should use slots for memory efficiency."""
        o = _make_outcome()
        assert hasattr(o, "__slots__")


# ═══════════════════════════════════════════════════════════════════
# record_outcome() tests
# ═══════════════════════════════════════════════════════════════════


class TestRecordOutcome:
    """Tests for DelegationGate.record_outcome()."""

    def test_record_outcome_increments_counters(self):
        """Recording an outcome should increment the appropriate counters."""
        gate = DelegationGate(calibrator=RecordingOverconfidentCalibrator())
        gate.record_outcome(_make_outcome(actual_success=True))
        assert gate._total_outcomes_recorded == 1
        assert gate._total_successful_delegations == 1
        assert gate._total_failed_delegations == 0

    def test_record_outcome_failure(self):
        """Recording a failed outcome should increment failure counter."""
        gate = DelegationGate(calibrator=RecordingOverconfidentCalibrator())
        gate.record_outcome(_make_outcome(actual_success=False))
        assert gate._total_outcomes_recorded == 1
        assert gate._total_successful_delegations == 0
        assert gate._total_failed_delegations == 1

    def test_record_outcome_feeds_calibrator(self):
        """When feedback_to_calibrator=True, the calibrator should receive the pair."""
        cal = RecordingOverconfidentCalibrator()
        gate = DelegationGate(calibrator=cal)
        gate.record_outcome(_make_outcome(
            predicted_confidence=0.6,
            actual_success=True,
            feedback_to_calibrator=True,
        ))
        assert len(cal.recorded) == 1
        assert cal.recorded[0] == (0.6, True)
        assert gate._total_calibrator_feedbacks == 1

    def test_record_outcome_skips_calibrator_feedback(self):
        """When feedback_to_calibrator=False, the calibrator should NOT receive the pair."""
        cal = RecordingOverconfidentCalibrator()
        gate = DelegationGate(calibrator=cal)
        gate.record_outcome(_make_outcome(
            predicted_confidence=0.6,
            actual_success=True,
            feedback_to_calibrator=False,
        ))
        assert len(cal.recorded) == 0
        assert gate._total_calibrator_feedbacks == 0

    def test_record_outcome_no_calibrator(self):
        """When no calibrator is attached, record_outcome should not raise."""
        gate = DelegationGate()  # No calibrator
        gate.record_outcome(_make_outcome(actual_success=True))
        assert gate._total_outcomes_recorded == 1
        assert gate._total_calibrator_feedbacks == 0

    def test_record_outcome_no_calibrator_feedback_flag(self):
        """With no calibrator but feedback=True, should not raise."""
        gate = DelegationGate()  # No calibrator
        gate.record_outcome(_make_outcome(
            actual_success=True,
            feedback_to_calibrator=True,
        ))
        assert gate._total_outcomes_recorded == 1
        # Should not crash; calibrator feedback silently skipped

    def test_record_multiple_outcomes(self):
        """Multiple outcomes should accumulate correctly."""
        cal = RecordingOverconfidentCalibrator()
        gate = DelegationGate(calibrator=cal)
        for success in [True, True, False, True, False]:
            gate.record_outcome(_make_outcome(actual_success=success))
        assert gate._total_outcomes_recorded == 5
        assert gate._total_successful_delegations == 3
        assert gate._total_failed_delegations == 2
        assert len(cal.recorded) == 5

    def test_outcomes_property_immutable(self):
        """The outcomes property should return an immutable tuple."""
        gate = DelegationGate()
        gate.record_outcome(_make_outcome())
        outcomes = gate.outcomes
        assert isinstance(outcomes, tuple)
        assert len(outcomes) == 1


# ═══════════════════════════════════════════════════════════════════
# Delegation accuracy tests
# ═══════════════════════════════════════════════════════════════════


class TestDelegationAccuracy:
    """Tests for the delegation_accuracy property."""

    def test_accuracy_zero_when_no_outcomes(self):
        """delegation_accuracy should return 0.0 when no outcomes recorded."""
        gate = DelegationGate()
        assert gate.delegation_accuracy == 0.0

    def test_accuracy_all_successful(self):
        """100% success rate should give accuracy 1.0."""
        gate = DelegationGate()
        for _ in range(5):
            gate.record_outcome(_make_outcome(actual_success=True))
        assert gate.delegation_accuracy == 1.0

    def test_accuracy_all_failed(self):
        """0% success rate should give accuracy 0.0."""
        gate = DelegationGate()
        for _ in range(5):
            gate.record_outcome(_make_outcome(actual_success=False))
        assert gate.delegation_accuracy == 0.0

    def test_accuracy_mixed(self):
        """3 successes out of 5 should give accuracy 0.6."""
        gate = DelegationGate(outcome_window=50)
        for success in [True, True, False, True, False]:
            gate.record_outcome(_make_outcome(actual_success=success))
        assert gate.delegation_accuracy == pytest.approx(0.6)

    def test_accuracy_rolling_window(self):
        """Accuracy should only use outcomes within the rolling window."""
        gate = DelegationGate(outcome_window=5)
        # Record 5 successes
        for _ in range(5):
            gate.record_outcome(_make_outcome(actual_success=True))
        assert gate.delegation_accuracy == 1.0
        # Record 5 failures — window should now contain only failures
        for _ in range(5):
            gate.record_outcome(_make_outcome(actual_success=False))
        assert gate.delegation_accuracy == 0.0

    def test_accuracy_rolling_window_partial(self):
        """Rolling window should slide correctly with mixed data."""
        gate = DelegationGate(outcome_window=4)
        # Record: S, S, F, S, F
        for success in [True, True, False, True, False]:
            gate.record_outcome(_make_outcome(actual_success=success))
        # Window should contain last 4: S, F, S, F → 2/4 = 0.5
        assert gate.delegation_accuracy == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════════
# Threshold adaptation tests
# ═══════════════════════════════════════════════════════════════════


class TestThresholdAdaptation:
    """Tests for dynamic ECE threshold adaptation based on delegation accuracy."""

    def test_no_adaptation_with_few_outcomes(self):
        """With < 3 outcomes, threshold should not adapt."""
        gate = DelegationGate(
            calibrator=RecordingOverconfidentCalibrator(),
            ece_threshold=0.15,
            adaptation_rate=0.02,
        )
        gate.record_outcome(_make_outcome(actual_success=True))
        gate.record_outcome(_make_outcome(actual_success=True))
        assert gate.ece_threshold == 0.15
        assert gate._threshold_adaptations == 0

    def test_tighten_threshold_high_accuracy(self):
        """High delegation accuracy (>0.8) should tighten (decrease) the threshold.

        When most delegated actions succeed, the system is being too
        cautious and should delegate less. Lowering the threshold means
        fewer actions exceed it, so fewer are delegated.
        """
        gate = DelegationGate(
            calibrator=RecordingOverconfidentCalibrator(),
            ece_threshold=0.15,
            adaptation_rate=0.02,
            outcome_window=50,
        )
        initial = gate.ece_threshold
        # Record 4 successful outcomes (accuracy = 1.0 > 0.8)
        for _ in range(4):
            gate.record_outcome(_make_outcome(actual_success=True))
        assert gate.ece_threshold < initial
        assert gate._threshold_adaptations >= 1

    def test_loosen_threshold_low_accuracy(self):
        """Low delegation accuracy (<0.5) should loosen (increase) the threshold.

        When most delegated actions fail, the system needs to delegate
        more aggressively. Raising the threshold means more actions
        exceed it, so more are delegated.
        """
        gate = DelegationGate(
            calibrator=RecordingOverconfidentCalibrator(),
            ece_threshold=0.15,
            adaptation_rate=0.02,
            outcome_window=50,
        )
        initial = gate.ece_threshold
        # Record 4 failed outcomes (accuracy = 0.0 < 0.5)
        for _ in range(4):
            gate.record_outcome(_make_outcome(actual_success=False))
        assert gate.ece_threshold > initial
        assert gate._threshold_adaptations >= 1

    def test_no_adaptation_moderate_accuracy(self):
        """Moderate accuracy (0.5 ≤ acc ≤ 0.8) should not change the threshold.

        The sweet spot — delegation is working reasonably well.
        """
        gate = DelegationGate(
            calibrator=RecordingOverconfidentCalibrator(),
            ece_threshold=0.15,
            adaptation_rate=0.02,
            outcome_window=50,
        )
        initial = gate.ece_threshold
        # 2 success, 2 failure → accuracy = 0.5 (boundary, in range)
        gate.record_outcome(_make_outcome(actual_success=True))
        gate.record_outcome(_make_outcome(actual_success=True))
        gate.record_outcome(_make_outcome(actual_success=False))
        gate.record_outcome(_make_outcome(actual_success=False))
        assert gate.ece_threshold == initial
        assert gate._threshold_adaptations == 0

    def test_threshold_has_floor(self):
        """Threshold should never drop below 0.05."""
        gate = DelegationGate(
            calibrator=RecordingOverconfidentCalibrator(),
            ece_threshold=0.051,  # Just above floor
            adaptation_rate=0.5,  # Very aggressive rate
            outcome_window=50,
        )
        # Record many successes to push threshold down
        for _ in range(10):
            gate.record_outcome(_make_outcome(actual_success=True))
        assert gate.ece_threshold >= 0.05

    def test_threshold_has_ceiling(self):
        """Threshold should never exceed 0.5."""
        gate = DelegationGate(
            calibrator=RecordingOverconfidentCalibrator(),
            ece_threshold=0.49,  # Just below ceiling
            adaptation_rate=0.5,  # Very aggressive rate
            outcome_window=50,
        )
        # Record many failures to push threshold up
        for _ in range(10):
            gate.record_outcome(_make_outcome(actual_success=False))
        assert gate.ece_threshold <= 0.5

    def test_adaptation_rate_property(self):
        """adaptation_rate should be accessible."""
        gate = DelegationGate(adaptation_rate=0.05)
        assert gate.adaptation_rate == 0.05

    def test_adaptation_is_conservative(self):
        """With default adaptation_rate=0.02, changes should be small."""
        gate = DelegationGate(
            calibrator=RecordingOverconfidentCalibrator(),
            ece_threshold=0.15,
            adaptation_rate=0.02,
            outcome_window=50,
        )
        initial = gate.ece_threshold
        # Record one batch of successful outcomes
        for _ in range(4):
            gate.record_outcome(_make_outcome(actual_success=True))
        # Change should be small: threshold * (1 - 0.02) per adaptation
        change = abs(gate.ece_threshold - initial)
        assert change < initial * 0.1  # Less than 10% change

    def test_reset_threshold(self):
        """reset_threshold() should restore the initial ECE threshold."""
        gate = DelegationGate(
            calibrator=RecordingOverconfidentCalibrator(),
            ece_threshold=0.15,
        )
        # Push threshold away from initial
        for _ in range(10):
            gate.record_outcome(_make_outcome(actual_success=True))
        assert gate.ece_threshold != 0.15
        # Reset
        gate.reset_threshold()
        assert gate.ece_threshold == 0.15


# ═══════════════════════════════════════════════════════════════════
# Serialization round-trip tests
# ═══════════════════════════════════════════════════════════════════


class TestOutcomeSerialization:
    """Tests for serialization/deserialization of outcomes."""

    def test_to_dict_includes_outcomes(self):
        """to_dict() should include outcome data."""
        gate = DelegationGate(calibrator=RecordingOverconfidentCalibrator())
        gate.record_outcome(_make_outcome(
            actual_success=True,
            predicted_confidence=0.6,
        ))
        data = gate.to_dict()
        assert "outcomes" in data
        assert len(data["outcomes"]) == 1
        assert data["outcomes"][0]["actual_success"] is True
        assert data["outcomes"][0]["predicted_confidence"] == 0.6

    def test_to_dict_includes_outcome_stats(self):
        """to_dict() should include outcome tracking stats."""
        gate = DelegationGate(calibrator=RecordingOverconfidentCalibrator())
        gate.record_outcome(_make_outcome(actual_success=True))
        data = gate.to_dict()
        assert data["total_outcomes_recorded"] == 1
        assert data["total_successful_delegations"] == 1
        assert data["total_calibrator_feedbacks"] == 1

    def test_to_dict_includes_adaptation_config(self):
        """to_dict() should include adaptation config."""
        gate = DelegationGate(adaptation_rate=0.05, outcome_window=100)
        data = gate.to_dict()
        assert data["adaptation_rate"] == 0.05
        assert data["outcome_window"] == 100
        assert "ece_threshold_initial" in data

    def test_round_trip_preserves_outcomes(self):
        """from_dict(to_dict()) should preserve recorded outcomes."""
        cal = RecordingOverconfidentCalibrator()
        gate = DelegationGate(calibrator=cal, ece_threshold=0.20)
        gate.record_outcome(_make_outcome(actual_success=True, predicted_confidence=0.7))
        gate.record_outcome(_make_outcome(actual_success=False, predicted_confidence=0.4))

        data = gate.to_dict()
        restored = DelegationGate.from_dict(data, calibrator=RecordingOverconfidentCalibrator())

        assert len(restored._outcomes) == 2
        assert restored._outcomes[0].actual_success is True
        assert restored._outcomes[0].predicted_confidence == 0.7
        assert restored._outcomes[1].actual_success is False
        assert restored._outcomes[1].predicted_confidence == 0.4

    def test_round_trip_preserves_outcome_stats(self):
        """from_dict(to_dict()) should preserve outcome counters."""
        cal = RecordingOverconfidentCalibrator()
        gate = DelegationGate(calibrator=cal)
        gate.record_outcome(_make_outcome(actual_success=True))
        gate.record_outcome(_make_outcome(actual_success=False))

        data = gate.to_dict()
        restored = DelegationGate.from_dict(data, calibrator=RecordingOverconfidentCalibrator())

        assert restored._total_outcomes_recorded == 2
        assert restored._total_successful_delegations == 1
        assert restored._total_failed_delegations == 1
        assert restored._total_calibrator_feedbacks == 2

    def test_round_trip_preserves_adapted_threshold(self):
        """from_dict(to_dict()) should preserve the adapted threshold."""
        cal = RecordingOverconfidentCalibrator()
        gate = DelegationGate(calibrator=cal, ece_threshold=0.15)
        # Push threshold down
        for _ in range(10):
            gate.record_outcome(_make_outcome(actual_success=True))
        adapted = gate.ece_threshold

        data = gate.to_dict()
        restored = DelegationGate.from_dict(data, calibrator=RecordingOverconfidentCalibrator())

        assert restored.ece_threshold == pytest.approx(adapted)
        assert restored._ece_threshold_initial == 0.15

    def test_round_trip_malformed_outcome_skipped(self):
        """from_dict() should skip malformed outcome entries."""
        data = {
            "outcomes": [
                {"action_id": "not-a-uuid", "action_description": "bad"},
            ],
        }
        gate = DelegationGate.from_dict(data)
        assert len(gate._outcomes) == 0

    def test_round_trip_empty_outcomes(self):
        """from_dict() with no outcomes should produce an empty list."""
        data: dict = {"outcomes": []}
        gate = DelegationGate.from_dict(data)
        assert len(gate._outcomes) == 0


# ═══════════════════════════════════════════════════════════════════
# Stats integration tests
# ═══════════════════════════════════════════════════════════════════


class TestOutcomeStats:
    """Tests for outcome tracking fields in the stats dict."""

    def test_stats_includes_outcome_fields(self):
        """stats dict should include all outcome tracking fields."""
        gate = DelegationGate(calibrator=RecordingOverconfidentCalibrator())
        gate.record_outcome(_make_outcome(actual_success=True))
        s = gate.stats
        assert "total_outcomes_recorded" in s
        assert "total_successful_delegations" in s
        assert "total_failed_delegations" in s
        assert "total_calibrator_feedbacks" in s
        assert "delegation_accuracy" in s
        assert "threshold_adaptations" in s
        assert "adaptation_rate" in s
        assert "outcome_window" in s
        assert "ece_threshold_initial" in s

    def test_stats_accuracy_matches_property(self):
        """stats['delegation_accuracy'] should be a rounded version of the property."""
        gate = DelegationGate()
        for success in [True, True, False]:
            gate.record_outcome(_make_outcome(actual_success=success))
        s = gate.stats
        # stats rounds to 4 decimal places
        expected = round(gate.delegation_accuracy, 4)
        assert s["delegation_accuracy"] == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════
# Repr tests
# ═══════════════════════════════════════════════════════════════════


class TestOutcomeRepr:
    """Tests for DelegationGate.__repr__ with outcome data."""

    def test_repr_includes_outcome_info(self):
        """repr should include outcome count, accuracy, and threshold."""
        gate = DelegationGate(calibrator=RecordingOverconfidentCalibrator())
        gate.record_outcome(_make_outcome(actual_success=True))
        r = repr(gate)
        assert "outcomes=1" in r
        assert "accuracy=" in r
        assert "threshold=" in r


# ═══════════════════════════════════════════════════════════════════
# Edge case: delegation outcome with ActionOrchestrator
# ═══════════════════════════════════════════════════════════════════


class TestOutcomeWithOrchestrator:
    """Integration tests for outcome tracking with ActionOrchestrator."""

    def test_orchestrator_to_dict_includes_outcome_data(self):
        """ActionOrchestrator.to_dict() should preserve outcome data."""
        cal = RecordingOverconfidentCalibrator()
        orch = ActionOrchestrator()
        gate = DelegationGate(calibrator=cal)
        orch.set_delegation_gate(gate=gate)
        # Execute some actions to generate delegation decisions
        for risk in [ActionRisk.LOW, ActionRisk.MODERATE, ActionRisk.HIGH]:
            orch.register_action(Action(description=f"t-{risk.name}", tool_name="t", risk=risk))
        orch.set_tension_profile({"autonomy_safety": 1.0, "sequential_parallel": 0.0, "verify_execute": 0.0})
        orch.execute_batch(lambda a: "ok")
        # Record outcomes for delegated actions
        for record in gate.records:
            if record.decision == DelegationDecision.DELEGATE:
                gate.record_outcome(DelegationOutcome(
                    action_id=record.action_id,
                    action_description=record.action_description,
                    action_risk=record.action_risk,
                    predicted_confidence=0.6,
                    actual_success=True,
                    delegated_mode=record.mode,
                    ece_at_delegation=record.ece,
                ))

        data = orch.to_dict()
        assert "delegation_gate" in data
        assert data["delegation_gate"]["total_outcomes_recorded"] >= 1
