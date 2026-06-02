"""Tests for calibration-driven verification depth modulation.

Iteration 007: When the ConfidenceCalibrator shows the agent is poorly
calibrated (high ECE), the Praxis orchestrator should increase verification
depth. When well-calibrated (low ECE), it can verify less — the agent has
learned its confidence estimates are trustworthy.
"""

import pytest

from isonome.praxis.orchestrator import (
    Action,
    ActionOrchestrator,
    ActionRisk,
    ExecutionReport,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def echo_executor():
    def executor(action):
        return {"echo": action.description}
    return executor


@pytest.fixture
def always_pass_validator():
    def validator(action, output):
        return (True, 1.0)
    return validator


@pytest.fixture
def overconfident_calibrator():
    """Calibrator pretrained to be heavily overconfident — high ECE."""
    from isonome.cognition.reasoning import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    for _ in range(50):
        cal.record(predicted_confidence=0.85, actual_success=False)
    for _ in range(20):
        cal.record(predicted_confidence=0.85, actual_success=True)
    # This gives high ECE — predicted 0.85 but actual accuracy ~0.29
    return cal


@pytest.fixture
def well_calibrated_calibrator():
    """Calibrator pretrained to be well-calibrated — low ECE."""
    from isonome.cognition.reasoning import ConfidenceCalibrator
    cal = ConfidenceCalibrator()
    for _ in range(50):
        cal.record(predicted_confidence=0.50, actual_success=True)
    for _ in range(50):
        cal.record(predicted_confidence=0.50, actual_success=False)
    for _ in range(30):
        cal.record(predicted_confidence=0.85, actual_success=True)
    for _ in range(6):
        cal.record(predicted_confidence=0.85, actual_success=False)
    cal.adjust_weights()
    return cal


# ═══════════════════════════════════════════════════════════════════
# ExecutionReport fields
# ═══════════════════════════════════════════════════════════════════


class TestReportFields:
    """New calibration-driven verify fields on ExecutionReport."""

    def test_report_has_calibration_ece_field(self):
        r = ExecutionReport(
            actions_total=1, actions_completed=1, actions_failed=0,
            actions_blocked=0, actions_retried=0, total_duration_ms=10,
            success_rate=1.0, avg_validation_score=0.5,
            parallelism_level=1, gate_blocks=0, tension_profile={},
        )
        assert r.calibration_ece == 0.0
        assert r.verify_modulation == 0.0

    def test_report_accepts_custom_values(self):
        r = ExecutionReport(
            actions_total=1, actions_completed=1, actions_failed=0,
            actions_blocked=0, actions_retried=0, total_duration_ms=10,
            success_rate=1.0, avg_validation_score=0.5,
            parallelism_level=1, gate_blocks=0, tension_profile={},
            calibration_ece=0.2345, verify_modulation=0.15,
        )
        assert r.calibration_ece == 0.2345
        assert r.verify_modulation == 0.15


# ═══════════════════════════════════════════════════════════════════
# No calibrator — default behavior unchanged
# ═══════════════════════════════════════════════════════════════════


class TestNoCalibratorVerifyDepth:
    """When no calibrator is set, verify depth is unchanged from baseline."""

    def test_no_calibrator_verify_is_zero_modulation(self, echo_executor):
        orch = ActionOrchestrator()
        orch.register_action(Action(description="task", tool_name="echo"))
        report = orch.execute_batch(echo_executor)
        assert report.calibration_applied is False
        assert report.calibration_ece == 0.0
        assert report.verify_modulation == 0.0

    def test_no_calibrator_verify_tension_unchanged(self, echo_executor):
        orch = ActionOrchestrator()
        orch.set_tension_profile({"verify_execute": -0.5})  # verify_heavy
        orch.register_action(Action(description="t", tool_name="echo"))
        report = orch.execute_batch(echo_executor)
        assert report.verify_modulation == 0.0


# ═══════════════════════════════════════════════════════════════════
# Overconfident calibrator → high ECE → increased verification
# ═══════════════════════════════════════════════════════════════════


class TestOverconfidentVerifyBoost:
    """When calibrator is overconfident (high ECE), verify depth increases."""

    def test_high_ece_increases_verify_modulation(
        self, echo_executor, overconfident_calibrator
    ):
        orch = ActionOrchestrator(
            confidence_calibrator=overconfident_calibrator,
        )
        orch.register_action(Action(description="t", tool_name="echo"))
        report = orch.execute_batch(echo_executor)
        assert report.calibration_applied is True
        assert report.calibration_ece > 0.10  # High ECE
        assert report.verify_modulation > 0.0  # Positive modulation

    def test_high_ece_report_summary_consistent(
        self, echo_executor, overconfident_calibrator
    ):
        orch = ActionOrchestrator(
            confidence_calibrator=overconfident_calibrator,
        )
        orch.register_action(Action(description="t", tool_name="echo"))
        report = orch.execute_batch(echo_executor)
        assert report.calibration_ece >= report.verify_modulation / 2.0
