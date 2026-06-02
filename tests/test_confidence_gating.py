
import pytest

from isonome.praxis.orchestrator import (
    Action,
    ActionOrchestrator,
    ActionRisk,
    ExecutionReport,
)
from isonome.praxis.pillar import PraxisPillar


# ═══════════════════════════════════════════════════════════════════
# Confidence-based safety gating tests
# ═══════════════════════════════════════════════════════════════════


class TestConfidenceSafetyGating:
    """Tests for calibrated confidence-based safety gating in the orchestrator."""

    @pytest.fixture
    def calibrator(self):
        """Return a ConfidenceCalibrator pretrained to be overconfident."""
        from isonome.cognition.reasoning import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        for _ in range(30):
            cal.record(predicted_confidence=0.50, actual_success=True)
        for _ in range(20):
            cal.record(predicted_confidence=0.50, actual_success=False)
        for _ in range(30):
            cal.record(predicted_confidence=0.85, actual_success=True)
        for _ in range(55):
            cal.record(predicted_confidence=0.85, actual_success=False)
        return cal

    @pytest.fixture
    def well_calibrated(self):
        """Return a well-calibrated calibrator."""
        from isonome.cognition.reasoning import ConfidenceCalibrator
        cal = ConfidenceCalibrator()
        for _ in range(30):
            cal.record(predicted_confidence=0.50, actual_success=True)
        for _ in range(30):
            cal.record(predicted_confidence=0.50, actual_success=False)
        for _ in range(30):
            cal.record(predicted_confidence=0.85, actual_success=True)
        for _ in range(6):
            cal.record(predicted_confidence=0.85, actual_success=False)
        return cal

    @pytest.fixture
    def echo_executor(self):
        def executor(action):
            return {"echo": action.description}
        return executor

    # ── Action field ──────────────────────────────────────────

    def test_action_confidence_default(self):
        a = Action(description="test", tool_name="echo")
        assert a.confidence_required == 0.0

    def test_action_confidence_custom(self):
        a = Action(description="risky", tool_name="deploy",
                   confidence_required=0.85)
        assert a.confidence_required == 0.85

    # ── Orchestrator calibrator wiring ────────────────────────

    def test_orchestrator_accepts_calibrator(self, calibrator):
        orch = ActionOrchestrator(confidence_calibrator=calibrator)
        assert orch._confidence_calibrator is calibrator

    def test_orchestrator_default_no_calibrator(self):
        orch = ActionOrchestrator()
        assert orch._confidence_calibrator is None

    def test_set_confidence_calibrator(self, calibrator):
        orch = ActionOrchestrator()
        assert orch._confidence_calibrator is None
        orch.set_confidence_calibrator(calibrator)
        assert orch._confidence_calibrator is calibrator
        orch.set_confidence_calibrator(None)
        assert orch._confidence_calibrator is None

    # ── ExecutionReport fields ────────────────────────────────

    def test_report_default_fields(self):
        r = ExecutionReport(
            actions_total=1, actions_completed=1, actions_failed=0,
            actions_blocked=0, actions_retried=0, total_duration_ms=100.0,
            success_rate=1.0, avg_validation_score=0.9, parallelism_level=1,
            gate_blocks=0, tension_profile={},
        )
        assert r.confidence_blocks == 0
        assert r.calibration_applied is False

    def test_report_confidence_fields_explicit(self):
        r = ExecutionReport(
            actions_total=1, actions_completed=0, actions_failed=0,
            actions_blocked=1, actions_retried=0, total_duration_ms=100.0,
            success_rate=0.0, avg_validation_score=0.0, parallelism_level=1,
            gate_blocks=0, tension_profile={},
            confidence_blocks=1, calibration_applied=True,
        )
        assert r.confidence_blocks == 1
        assert r.calibration_applied is True

    # ── Confidence gating: no calibrator = no blocks ──────────

    def test_no_calibrator_no_confidence_blocks(self, echo_executor):
        orch = ActionOrchestrator()
        orch.register_action(Action(
            description="should pass", tool_name="echo",
            confidence_required=0.95,
        ))
        report = orch.execute_batch(executor_fn=echo_executor)
        assert report.actions_completed == 1
        assert report.actions_blocked == 0
        assert report.confidence_blocks == 0
        assert report.calibration_applied is False

    def test_no_calibrator_zero_confidence_no_block(self, echo_executor):
        orch = ActionOrchestrator()
        orch.register_action(Action(
            description="zero conf", tool_name="echo",
            confidence_required=0.0,
        ))
        report = orch.execute_batch(executor_fn=echo_executor)
        assert report.actions_completed == 1
        assert report.confidence_blocks == 0

    # ── Confidence gating: with calibrator ────────────────────

    def test_confidence_blocks_low_action(self, calibrator, echo_executor):
        orch = ActionOrchestrator(confidence_calibrator=calibrator)
        # Use safe mode so confidence threshold is high (theta = 0.78)
        orch.set_tension_profile({"autonomy_safety": -0.4})
        # Use LOW risk so the risk gate does NOT block (risk_q=0.25, tau=0.3)
        # The confidence gate then evaluates calibrated_confidence(0.85) ~ 0.35
        # 0.35 < 0.78 -> BLOCKED by confidence gate
        orch.register_action(Action(
            description="risky deploy", tool_name="deploy",
            confidence_required=0.85,
            risk=ActionRisk.LOW,
        ))
        report = orch.execute_batch(executor_fn=echo_executor)
        assert report.actions_blocked == 1  # Blocked by confidence gate
        assert report.confidence_blocks == 1
        assert report.calibration_applied is True

    def test_high_autonomy_lowers_confidence_threshold(self, calibrator, echo_executor):
        orch = ActionOrchestrator(confidence_calibrator=calibrator)
        orch.set_tension_profile({"autonomy_safety": 1.0})
        orch.register_action(Action(
            description="deploy", tool_name="deploy",
            confidence_required=0.50,
        ))
        report = orch.execute_batch(executor_fn=echo_executor)
        assert report.actions_completed == 1
        assert report.confidence_blocks == 0
        assert report.calibration_applied is True

    def test_safe_mode_high_threshold_blocks(self, calibrator, echo_executor):
        orch = ActionOrchestrator(confidence_calibrator=calibrator)
        # Max safe: tau=0.0. Only TRIVIAL passes risk gate.
        # Use TRIVIAL risk + confidence_required=0.85:
        #   risk gate: risk_q=0.0 <= tau=0.0 -> passes
        #   confidence gate: theta=0.9, calibrated(0.85)~0.35 < 0.9 -> blocked
        orch.set_tension_profile({"autonomy_safety": -1.0})
        orch.register_action(Action(
            description="deploy", tool_name="deploy",
            confidence_required=0.85,
            risk=ActionRisk.TRIVIAL,
        ))
        report = orch.execute_batch(executor_fn=echo_executor)
        assert report.actions_blocked == 1  # Blocked by confidence gate
        assert report.confidence_blocks == 1

    def test_well_calibrated_confidence_passes(self, well_calibrated, echo_executor):
        orch = ActionOrchestrator(confidence_calibrator=well_calibrated)
        orch.set_tension_profile({"autonomy_safety": 0.0})
        orch.register_action(Action(
            description="confident task", tool_name="echo",
            confidence_required=0.85,
        ))
        report = orch.execute_batch(executor_fn=echo_executor)
        assert report.actions_completed == 1
        assert report.confidence_blocks == 0

    def test_confidence_gating_with_approval(self, calibrator, echo_executor):
        orch = ActionOrchestrator(confidence_calibrator=calibrator)
        orch.set_tension_profile({"autonomy_safety": -0.4})
        orch.register_action(Action(
            description="approved deploy", tool_name="deploy",
            confidence_required=0.85,
            risk=ActionRisk.MODERATE,
        ))
        report = orch.execute_batch(
            executor_fn=echo_executor,
            approve_fn=lambda a: True,
        )
        assert report.actions_completed == 1
        assert report.actions_blocked == 0
        assert report.confidence_blocks == 0

    def test_confidence_no_effect_on_zero_conf_actions(self, calibrator, echo_executor):
        orch = ActionOrchestrator(confidence_calibrator=calibrator)
        # Neutral autonomy so both LOW and CRITICAL pass the risk gate
        # (tau = 0.5, risk_q LOW = 0.25, risk_q CRITICAL = 1.0 -> CRITICAL blocked)
        orch.set_tension_profile({"autonomy_safety": 0.0})
        orch.register_action(Action(
            description="normal", tool_name="echo",
            risk=ActionRisk.LOW,
        ))
        orch.register_action(Action(
            description="critical", tool_name="rm",
            risk=ActionRisk.CRITICAL,
        ))
        report = orch.execute_batch(executor_fn=echo_executor)
        # LOW passes risk gate (0.25 <= 0.5), CRITICAL blocked by risk gate (1.0 > 0.5)
        # Neither has confidence_required, so confidence_blocks=0
        assert report.actions_completed == 1
        assert report.actions_blocked == 1  # CRITICAL blocked by risk
        assert report.confidence_blocks == 0  # No confidence blocking

    # ── import_from_cognition carries confidence ──────────────

    def test_import_carries_confidence(self):
        orch = ActionOrchestrator()
        tasks = [
            {
                "description": "deploy app",
                "tool_name": "deploy",
                "risk": "HIGH",
                "confidence_required": 0.92,
            },
            {
                "description": "check logs",
                "tool_name": "read",
                "confidence_required": 0.0,
            },
        ]
        ids = orch.import_from_cognition(tasks)
        assert len(ids) == 2
        actions = list(orch._actions.values())
        assert actions[0].confidence_required == 0.92
        assert actions[1].confidence_required == 0.0

    def test_import_defaults_confidence_zero(self):
        orch = ActionOrchestrator()
        tasks = [{"description": "simple", "tool_name": "echo"}]
        ids = orch.import_from_cognition(tasks)
        action = list(orch._actions.values())[0]
        assert action.confidence_required == 0.0

    # ── Serialization round-trip ───────────────────────────────

    def test_serialization_preserves_confidence(self, calibrator):
        orch = ActionOrchestrator(confidence_calibrator=calibrator)
        orch.register_action(Action(
            description="confident", tool_name="echo",
            confidence_required=0.88,
        ))
        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)
        actions = list(restored._actions.values())
        assert len(actions) == 1
        assert actions[0].confidence_required == 0.88
        assert restored._confidence_calibrator is None

    # ── PraxisPillar calibrator wiring ────────────────────────

    def test_pillar_accepts_calibrator(self, calibrator):
        pp = PraxisPillar(name="test", confidence_calibrator=calibrator)
        assert pp._confidence_calibrator is calibrator

    def test_pillar_passes_to_orchestrator(self, calibrator):
        from isonome.types import AgentIdentity, AgentState
        state = AgentState(identity=AgentIdentity(name="test"))
        pp = PraxisPillar(name="test", confidence_calibrator=calibrator)
        pp.initialize(state)
        assert pp.orchestrator is not None
        assert pp.orchestrator._confidence_calibrator is calibrator

    def test_pillar_set_calibrator_after_init(self, calibrator):
        from isonome.types import AgentIdentity, AgentState
        state = AgentState(identity=AgentIdentity(name="test"))
        pp = PraxisPillar(name="test")
        pp.initialize(state)
        assert pp._confidence_calibrator is None
        pp.set_confidence_calibrator(calibrator)
        assert pp._confidence_calibrator is calibrator
        assert pp.orchestrator._confidence_calibrator is calibrator

    def test_pillar_set_to_none(self, calibrator):
        from isonome.types import AgentIdentity, AgentState
        state = AgentState(identity=AgentIdentity(name="test"))
        pp = PraxisPillar(name="test", confidence_calibrator=calibrator)
        pp.initialize(state)
        pp.set_confidence_calibrator(None)
        assert pp._confidence_calibrator is None
        assert pp.orchestrator._confidence_calibrator is None

    def test_pillar_e2e_gating(self, calibrator, echo_executor):
        from isonome.types import AgentIdentity, AgentState
        state = AgentState(identity=AgentIdentity(name="test"))
        pp = PraxisPillar(
            name="executor",
            executor_fn=echo_executor,
            confidence_calibrator=calibrator,
        )
        pp.initialize(state)
        pp.update_tension_profile({"autonomy_safety": -0.4})
        pp.orchestrator.register_action(Action(
            description="risky deploy", tool_name="deploy",
            confidence_required=0.85,
        ))
        report = pp.execute_pending()
        assert report is not None
        assert report.confidence_blocks >= 1
        assert report.calibration_applied is True

    # ── Edge cases ─────────────────────────────────────────────

    def test_confidence_threshold_bounded(self, calibrator):
        orch = ActionOrchestrator(confidence_calibrator=calibrator)
        orch.set_tension_profile({"autonomy_safety": -10.0})
        orch.register_action(Action(
            description="deploy", tool_name="deploy",
            confidence_required=0.99,
        ))
        report = orch.execute_batch(executor_fn=lambda a: {"ok": True})
        assert report.calibration_applied is True
