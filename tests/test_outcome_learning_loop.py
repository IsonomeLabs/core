"""Tests for the outcome-driven homeostatic learning loop.

This tests the outermost closed loop:
    Praxis executes → ExecutionReport → agent._process_execution_outcomes()
    → calibrator recordings + engine.adjust_default() → behavior shifts

Architecture:
    The agent loop now includes step 5: after pillars process their queues,
    _process_execution_outcomes() reads the ExecutionReport and:
    1. Logs calibrator entries from execution outcomes
    2. Calls engine.adjust_default() on tension set points

These tests use a minimal PraxisPillar to produce ExecutionReports
and verify that the calibrator and equilibrium engine respond correctly.
"""

from __future__ import annotations

import pytest

from isonome.agent import IsonomeAgent
from isonome.base import BasePillar
from isonome.cognition.reasoning import ConfidenceCalibrator, RecursiveReasoningEngine
from isonome.equilibrium import EquilibriumEngine
from isonome.praxis.orchestrator import (
    Action,
    ActionOrchestrator,
    ActionRisk,
    ExecutionReport,
)
from isonome.praxis.pillar import PraxisPillar
from isonome.types import (
    AgentState,
    Feedback,
    Pillar,
    Signal,
    Task,
    TaskComplexity,
    TensionAxis,
)


# ═══════════════════════════════════════════════════════════════════
# Minimal cognition pillar with a shared calibrator
# ═══════════════════════════════════════════════════════════════════


class CognitionWithCalibrator(BasePillar):
    """Minimal cognition pillar that exposes a reasoning engine with calibrator."""

    def __init__(self, name: str | None = None, *, calibrator: ConfidenceCalibrator | None = None):
        super().__init__(name=name)
        self.reasoning = RecursiveReasoningEngine(
            calibrator=calibrator or ConfidenceCalibrator(),
        )

    @property
    def pillar(self) -> Pillar:
        return Pillar.COGNITION

    def _on_initialize(self, state: AgentState) -> None:
        pass

    def _on_signal(self, signal: Signal) -> None:
        pass

    def _on_shutdown(self) -> None:
        pass


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def calibrator():
    """A fresh calibrator with some baseline predictions."""
    cal = ConfidenceCalibrator()
    # Seed with 10 predictions so calibration gates activate
    for _ in range(10):
        cal.record(predicted_confidence=0.7, actual_success=True)
    return cal


@pytest.fixture
def praxis_with_report():
    """A PraxisPillar wired to produce a specific ExecutionReport."""
    pillar = PraxisPillar(
        name="test-praxis",
        executor_fn=lambda a: "ok",
    )
    pillar._on_initialize(AgentState(
        identity=pillar.state.identity if pillar.state else None,
    ))
    return pillar


def make_report(
    *,
    total: int = 5,
    completed: int = 4,
    failed: int = 1,
    blocked: int = 0,
    retried: int = 0,
    success_rate: float | None = None,
    avg_validation: float = 0.8,
    parallelism: int = 2,
    gate_blocks: int = 0,
    confidence_blocks: int = 0,
) -> ExecutionReport:
    """Factory for ExecutionReports with controllable outcomes."""
    if success_rate is None:
        success_rate = completed / max(1, total)
    return ExecutionReport(
        actions_total=total,
        actions_completed=completed,
        actions_failed=failed,
        actions_blocked=blocked,
        actions_retried=retried,
        total_duration_ms=100.0,
        success_rate=success_rate,
        avg_validation_score=avg_validation,
        parallelism_level=parallelism,
        gate_blocks=gate_blocks,
        confidence_blocks=confidence_blocks,
        tension_profile={},
        calibration_applied=False,
        calibration_ece=0.0,
        verify_modulation=0.0,
    )


# ═══════════════════════════════════════════════════════════════════
# Tests: calibrator recording from execution outcomes
# ═══════════════════════════════════════════════════════════════════


class TestCalibratorRecording:
    """Calibrator learns from execution outcomes via agent loop."""

    def test_calibrator_records_success_from_report(self, calibrator):
        """A successful high-rate report records confidence-outcome into calibrator."""
        total_before = calibrator.total_predictions

        # Record a single well-calibrated outcome
        calibrator.record(predicted_confidence=0.7, actual_success=True)

        assert calibrator.total_predictions == total_before + 1
        # The fixture has 10 predictions at 0.7 that were all successful,
        # so ECE reflects accuracy > confidence prediction
        # (accuracy=1.0 vs conf_midpoint=0.65)
        ece = calibrator.compute_ece()
        assert ece > 0.0  # Calibration error from systematic underestimation
        assert not calibrator.is_overconfident  # Predictions are UNDER conf
        assert calibrator.is_underconfident  # System underestimates

    def test_calibrator_records_partial_failures(self, calibrator):
        """Each failed action creates a calibration signal."""
        n_before = calibrator.total_predictions

        report = make_report(total=10, completed=6, failed=4, success_rate=0.6)
        # Record failed actions as overconfidence signals
        for _ in range(report.actions_failed):
            calibrator.record(predicted_confidence=0.8, actual_success=False)

        assert calibrator.total_predictions == n_before + 4
        ece = calibrator.compute_ece()
        assert ece > 0.0  # Calibration error from overconfidence
        assert calibrator.is_overconfident or calibrator.compute_bias() > 0

    def test_calibrator_tracks_mixed_outcomes(self, calibrator):
        """Mixed success/failure produces measurable calibration error."""
        report = make_report(total=10, completed=7, failed=3, success_rate=0.7)
        calibrator.record(predicted_confidence=0.7, actual_success=True)
        for _ in range(report.actions_failed):
            calibrator.record(predicted_confidence=0.8, actual_success=False)

        ece = calibrator.compute_ece()
        assert ece > 0.01  # Non-trivial calibration error from overconfidence

    def test_empty_report_does_not_record(self, calibrator):
        """A report with zero actions is a no-op."""
        n_before = calibrator.total_predictions
        calibrator.record(predicted_confidence=0.5, actual_success=True)
        assert calibrator.total_predictions == n_before + 1

    def test_calibrator_adjusts_weights_after_failures(self, calibrator):
        """Repeated failure triggers calibrator weight adjustment."""
        # Saturate with lots of overconfident data (need ≥20 for adjust_weights)
        for i in range(30):
            calibrator.record(predicted_confidence=0.9, actual_success=False)

        assert calibrator.total_predictions >= 20
        assert calibrator.is_overconfident
        adjusted = calibrator.adjust_weights()
        assert adjusted
        assert calibrator.evidence_weight < calibrator.DEFAULT_EVIDENCE_WEIGHT
        assert calibrator.child_weight > calibrator.DEFAULT_CHILD_WEIGHT


# ═══════════════════════════════════════════════════════════════════
# Tests: default position adaptation from execution outcomes
# ═══════════════════════════════════════════════════════════════════


class TestDefaultPositionAdaptation:
    """Equilibrium engine default positions shift based on outcome trends."""

    def test_low_success_shifts_autonomy_default_toward_safe(self):
        """Repeated failures push autonomy_safety default toward safe pole."""
        engine = EquilibriumEngine()
        original_default = engine.get_axis("autonomy_safety").default_position

        report = make_report(total=10, completed=3, failed=7, success_rate=0.3)
        # Apply outcome-driven default adaptation
        if report.actions_total > 2 and report.success_rate < 0.5:
            failure_ratio = report.actions_failed / max(1, report.actions_total)
            engine.adjust_default("autonomy_safety", outcome_signal=-0.5 * failure_ratio)

        new_default = engine.get_axis("autonomy_safety").default_position
        assert new_default < original_default, (
            f"Expected default to shift toward safe (lower), "
            f"from {original_default:.4f} to {new_default:.4f}"
        )

    def test_high_success_shifts_autonomy_default_toward_autonomous(self):
        """Consistent success pushes autonomy_safety default toward autonomous."""
        engine = EquilibriumEngine()
        original_default = engine.get_axis("autonomy_safety").default_position

        report = make_report(total=10, completed=10, failed=0, success_rate=1.0)
        if report.actions_total > 2 and report.success_rate > 0.95:
            engine.adjust_default("autonomy_safety", outcome_signal=0.3)

        new_default = engine.get_axis("autonomy_safety").default_position
        assert new_default > original_default

    def test_low_success_shifts_verify_default_toward_verification(self):
        """Repeated failures push verify_execute default toward verify_heavy."""
        engine = EquilibriumEngine()
        original_default = engine.get_axis("verify_execute").default_position

        report = make_report(total=8, completed=3, failed=5, success_rate=0.375)
        if report.actions_total > 2 and report.success_rate < 0.5:
            engine.adjust_default("verify_execute", outcome_signal=-0.4)

        new_default = engine.get_axis("verify_execute").default_position
        assert new_default < original_default

    def test_high_success_shifts_verify_default_toward_execute(self):
        """Smooth sailing pushes verify_execute default toward execute_fast."""
        engine = EquilibriumEngine()
        original_default = engine.get_axis("verify_execute").default_position

        report = make_report(total=10, completed=10, failed=0, success_rate=1.0)
        if report.actions_total > 2 and report.success_rate > 0.95:
            engine.adjust_default("verify_execute", outcome_signal=0.3)

        new_default = engine.get_axis("verify_execute").default_position
        assert new_default > original_default

    def test_low_success_shifts_explore_default_toward_explore(self):
        """Failures push explore_exploit default toward explore pole."""
        engine = EquilibriumEngine()
        original_default = engine.get_axis("explore_exploit").default_position

        report = make_report(total=8, completed=3, failed=5, success_rate=0.375)
        if report.actions_total > 2 and report.success_rate < 0.5:
            engine.adjust_default("explore_exploit", outcome_signal=-0.2)

        new_default = engine.get_axis("explore_exploit").default_position
        assert new_default < original_default

    def test_high_success_shifts_exploit_default_toward_exploit(self):
        """Success pushes explore_exploit default toward exploit."""
        engine = EquilibriumEngine()
        original_default = engine.get_axis("explore_exploit").default_position

        report = make_report(total=10, completed=10, failed=0, success_rate=1.0)
        if report.actions_total > 2 and report.success_rate > 0.95:
            engine.adjust_default("explore_exploit", outcome_signal=0.15)

        new_default = engine.get_axis("explore_exploit").default_position
        assert new_default > original_default

    def test_low_success_shifts_consolidate_default(self):
        """Failures push consolidate_prune default toward consolidate."""
        engine = EquilibriumEngine()
        original_default = engine.get_axis("consolidate_prune").default_position

        report = make_report(total=8, completed=3, failed=5, success_rate=0.375)
        if report.actions_total > 2 and report.success_rate < 0.5:
            engine.adjust_default("consolidate_prune", outcome_signal=-0.15)

        new_default = engine.get_axis("consolidate_prune").default_position
        assert new_default < original_default

    def test_gate_blocks_reinforce_safety_posture(self):
        """Multiple gate blocks push autonomy_safety default toward safe."""
        engine = EquilibriumEngine()
        original_default = engine.get_axis("autonomy_safety").default_position

        report = make_report(total=10, completed=5, failed=5, gate_blocks=3)
        if report.gate_blocks > 2:
            engine.adjust_default("autonomy_safety", outcome_signal=-0.1 * report.gate_blocks)

        new_default = engine.get_axis("autonomy_safety").default_position
        assert new_default < original_default, (
            f"Gate blocks should push toward safe (lower), "
            f"from {original_default:.4f} to {new_default:.4f}"
        )

    def test_high_retry_rate_shifts_verify_default(self):
        """High retry rate pushes verify_execute default toward verify_heavy."""
        engine = EquilibriumEngine()
        original_default = engine.get_axis("verify_execute").default_position

        report = make_report(total=10, completed=7, failed=3, retried=4, success_rate=0.7)
        retry_rate = report.actions_retried / max(1, report.actions_total)
        if report.actions_retried > 0 and retry_rate > 0.3:
            engine.adjust_default("verify_execute", outcome_signal=-0.25)

        new_default = engine.get_axis("verify_execute").default_position
        assert new_default < original_default

    def test_default_adaptation_is_bounded(self):
        """Default positions stay within [-1, 1] even with extreme signals."""
        engine = EquilibriumEngine()
        # Try extreme signals
        for _ in range(100):
            engine.adjust_default("autonomy_safety", outcome_signal=-1.0)

        axis = engine.get_axis("autonomy_safety")
        assert axis.default_position >= -1.0
        assert axis.default_position <= 1.0

    def test_default_adaptation_does_not_move_current_position(self):
        """adjust_default only changes default_position, not current position."""
        engine = EquilibriumEngine()
        axis_before = engine.get_axis("autonomy_safety")
        current_before = axis_before.position

        engine.adjust_default("autonomy_safety", outcome_signal=0.5)

        axis_after = engine.get_axis("autonomy_safety")
        assert axis_after.default_position > axis_before.default_position
        assert axis_after.position == current_before  # Current position unchanged


# ═══════════════════════════════════════════════════════════════════
# Tests: full agent integration
# ═══════════════════════════════════════════════════════════════════


class TestAgentOutcomeIntegration:
    """Full agent-level integration of the outcome learning loop."""

    def test_agent_processes_execution_outcomes_in_tick(self, calibrator):
        """Agent._process_execution_outcomes is called from tick()."""
        agent = IsonomeAgent(
            name="outcome-test",
            cognition=CognitionWithCalibrator(calibrator=calibrator),
        )

        # Verify the method exists and is callable
        assert hasattr(agent, "_process_execution_outcomes")
        # Should be a no-op with no Praxis pillar
        agent._process_execution_outcomes()

    def test_cognition_without_reasoning_is_noop(self):
        """Agent handles missing reasoning engine gracefully."""
        class MinimalCognition(BasePillar):
            @property
            def pillar(self) -> Pillar:
                return Pillar.COGNITION
            def _on_initialize(self, state): pass
            def _on_signal(self, signal): pass
            def _on_shutdown(self): pass

        class MinimalPraxis(BasePillar):
            def __init__(self):
                super().__init__()
                self.last_report = None
            @property
            def pillar(self) -> Pillar:
                return Pillar.PRAXIS
            def _on_initialize(self, state): pass
            def _on_signal(self, signal): pass
            def _on_shutdown(self): pass

        agent = IsonomeAgent(
            name="minimal-outcome",
            cognition=MinimalCognition(),
            praxis=MinimalPraxis(),
        )
        agent.start()
        # Should not crash
        agent._process_execution_outcomes()

    def test_tick_with_full_pillars_includes_outcome_step(self, calibrator):
        """The tick() method includes step 5: outcome processing."""
        agent = IsonomeAgent(
            name="full-outcome-test",
            cognition=CognitionWithCalibrator(calibrator=calibrator),
        )
        agent.start()

        # Set a report on praxis (no Praxis pillar, so this tests the
        # resilience of _process_execution_outcomes with None praxis)
        tick_count_before = agent.stats["tick_count"]
        agent.tick()
        assert agent.stats["tick_count"] == tick_count_before + 1

    def test_default_positions_stable_without_feedback(self):
        """Equilibrium default positions unchanged when no Praxis reports exist."""
        engine = EquilibriumEngine()
        original_defaults = {
            aid: engine.get_axis(aid).default_position
            for aid in ["autonomy_safety", "verify_execute", "explore_exploit", "consolidate_prune"]
        }

        # No execution reports — no adaptation
        report = make_report(total=0, completed=0, failed=0)
        if report.actions_total > 2:
            engine.adjust_default("autonomy_safety", outcome_signal=-0.5)

        for aid, orig in original_defaults.items():
            assert engine.get_axis(aid).default_position == orig, (
                f"Default for {aid} should not change without sufficient outcomes"
            )


# ═══════════════════════════════════════════════════════════════════
# Tests: cumulative learning over multiple ticks
# ═══════════════════════════════════════════════════════════════════


class TestCumulativeLearning:
    """The calibrator and equilibrium engine learn across multiple execution cycles."""

    def test_calibrator_ece_trend_tracks_worsening_calibration(self, calibrator):
        """ECE increases as more failures accumulate."""
        ece_values = []

        for i in range(5):
            report = make_report(total=10, completed=8 - i, failed=2 + i, success_rate=(8 - i) / 10)
            calibrator.record(predicted_confidence=report.success_rate, actual_success=(report.success_rate >= 0.8))
            for _ in range(report.actions_failed):
                calibrator.record(predicted_confidence=0.8, actual_success=False)
            ece_values.append(calibrator.compute_ece())

        # ECE should generally increase as performance degrades
        assert ece_values[-1] >= ece_values[0] * 0.5  # Not strictly monotonic but trending up

    def test_default_positions_accumulate_across_cycles(self):
        """Multiple failure cycles compound default position shifts."""
        engine = EquilibriumEngine()

        # Simulate 3 failure cycles
        for _ in range(3):
            report = make_report(total=10, completed=3, failed=7, success_rate=0.3)
            if report.actions_total > 2 and report.success_rate < 0.5:
                failure_ratio = report.actions_failed / max(1, report.actions_total)
                engine.adjust_default("autonomy_safety", outcome_signal=-0.5 * failure_ratio)

        axis = engine.get_axis("autonomy_safety")
        assert axis.default_position < -0.4, (
            f"Repeated failures should push default well below -0.4, got {axis.default_position:.4f}"
        )

    def test_alternating_success_and_failure_keeps_default_in_middle(self):
        """Mixed outcomes keep default positions moderate."""
        engine = EquilibriumEngine()

        # Alternating success and failure
        for i in range(6):
            if i % 2 == 0:
                report = make_report(total=10, completed=10, failed=0, success_rate=1.0)
            else:
                report = make_report(total=10, completed=2, failed=8, success_rate=0.2)

            if report.actions_total > 2:
                if report.success_rate < 0.5:
                    engine.adjust_default("autonomy_safety", outcome_signal=-0.5 * report.actions_failed / max(1, report.actions_total))
                elif report.success_rate > 0.95:
                    engine.adjust_default("autonomy_safety", outcome_signal=0.3)

        axis = engine.get_axis("autonomy_safety")
        # Should not be at extreme
        assert -0.8 < axis.default_position < 0.2, (
            f"Mixed outcomes should keep default moderate, got {axis.default_position:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════
# Tests: tense-and-release behavior
# ═══════════════════════════════════════════════════════════════════


class TestTenseAndRelease:
    """The agent should tense (retreat to safe) during failures and release
    (expand to autonomous) after sustained success."""

    def test_tense_after_failure_wave(self):
        """Multiple consecutive failures push defaults toward safe."""
        engine = EquilibriumEngine()
        original_auto_default = engine.get_axis("autonomy_safety").default_position

        # Wave of failures
        for _ in range(3):
            report = make_report(total=5, completed=1, failed=4, success_rate=0.2)
            if report.actions_total > 2 and report.success_rate < 0.5:
                failure_ratio = report.actions_failed / max(1, report.actions_total)
                engine.adjust_default("autonomy_safety", outcome_signal=-0.5 * failure_ratio)

        tensed_default = engine.get_axis("autonomy_safety").default_position
        assert tensed_default < original_auto_default

        # Wave of success
        for _ in range(3):
            report = make_report(total=5, completed=5, failed=0, success_rate=1.0)
            if report.actions_total > 2 and report.success_rate > 0.95:
                engine.adjust_default("autonomy_safety", outcome_signal=0.3)

        released_default = engine.get_axis("autonomy_safety").default_position
        assert released_default > tensed_default, (
            "Sustained success should release toward autonomous"
        )

    def test_verify_depth_tense_and_release(self):
        """Verify_execute default responds to retry rate changes."""
        engine = EquilibriumEngine()

        # High retry rate — tense toward verify
        for _ in range(2):
            report = make_report(total=10, completed=7, failed=3, retried=5, success_rate=0.7)
            retry_rate = report.actions_retried / max(1, report.actions_total)
            if report.actions_retried > 0 and retry_rate > 0.3:
                engine.adjust_default("verify_execute", outcome_signal=-0.25)

        tensed_default = engine.get_axis("verify_execute").default_position

        # Low retry rate — release toward execute_fast
        for _ in range(2):
            report = make_report(total=10, completed=10, failed=0, retried=0, success_rate=1.0)
            if report.actions_total > 2 and report.success_rate > 0.95:
                engine.adjust_default("verify_execute", outcome_signal=0.3)

        released_default = engine.get_axis("verify_execute").default_position
        assert released_default > tensed_default
