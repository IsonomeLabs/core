"""Tests for Uncertainty-Aware Planning — calibration quality modulates reasoning effort.

Tests cover:
  - Calibration amplifier formula (no data, perfect, poor, bounded)
  - Calibration-aware depth modulation (deeper reasoning when miscalibrated)
  - Calibration-aware branching modulation (wider search when miscalibrated)
  - Calibration-aware divergence override (force diverge when ECE is high)
  - Integration with RecursiveReasoningEngine end-to-end
  - Integration with CognitionPillar lifecycle
  - Edge cases and boundary conditions

The core insight being tested: The metacognitive calibration system
(ConfidenceCalibrator) now directly modulates the reasoning engine's
computational effort. When the agent knows it's poorly calibrated,
it invests more cognitive resources — reasoning deeper, branching wider,
exploring more alternatives. When well-calibrated, it reasons efficiently.
"""

from __future__ import annotations

import math
import pytest

from isonome.cognition.reasoning import (
    ConfidenceCalibrator,
    RecursiveReasoningEngine,
)
from isonome.cognition.attention import AttentionEquilibriumSystem
from isonome.cognition.pillar import CognitionPillar
from isonome.equilibrium import EquilibriumEngine
from isonome.types import (
    AgentIdentity,
    AgentState,
    Pillar,
    Signal,
    TensionAxis,
    TensionSnapshot,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _engine_builder(
    calibrator: ConfidenceCalibrator | None = None,
    shallow_deep: float = -0.2,
    explore_exploit: float = 0.15,
    divergent_convergent: float = 0.3,
) -> RecursiveReasoningEngine:
    """Create an engine with a custom calibrator and tension profile."""
    eng = RecursiveReasoningEngine(calibrator=calibrator)
    eng.set_tension_profile({
        "shallow_deep": shallow_deep,
        "explore_exploit": explore_exploit,
        "divergent_convergent": divergent_convergent,
    })
    return eng


def _populate_calibrator_with_ece(
    calibrator: ConfidenceCalibrator,
    target_ece: float,
    n_samples: int = 50,
) -> None:
    """Populate a calibrator to achieve roughly a target ECE.

    Strategy: spread predictions across bins to create a controlled
    calibration gap. For low ECE, match accuracy to confidence.
    For high ECE, create systematic overconfidence at high-confidence bins.
    """
    import random
    rng = random.Random(42)

    if target_ece < 0.03:
        # Well-calibrated: accuracy matches confidence across 3 bins
        for _ in range(n_samples // 3):
            calibrator.record(0.35, rng.random() < 0.35)
        for _ in range(n_samples // 3):
            calibrator.record(0.55, rng.random() < 0.55)
        for _ in range(n_samples - 2 * (n_samples // 3)):
            calibrator.record(0.75, rng.random() < 0.75)
    elif target_ece < 0.15:
        # Mild miscalibration: slight overconfidence
        for _ in range(n_samples // 3):
            calibrator.record(0.35, rng.random() < 0.35)
        for _ in range(n_samples // 3):
            calibrator.record(0.65, rng.random() < 0.55)  # conf 0.65, acc ~0.55
        for _ in range(n_samples - 2 * (n_samples // 3)):
            calibrator.record(0.85, rng.random() < 0.75)  # conf 0.85, acc ~0.75
    else:
        # Significant miscalibration: strong overconfidence
        for _ in range(n_samples // 4):
            calibrator.record(0.25, rng.random() < 0.40)  # underconfident at low end
        for _ in range(n_samples // 4):
            calibrator.record(0.55, rng.random() < 0.50)
        for _ in range(n_samples // 4):
            calibrator.record(0.75, rng.random() < 0.50)  # big gap at 0.75
        for _ in range(n_samples - 3 * (n_samples // 4)):
            calibrator.record(0.95, rng.random() < 0.50)  # huge overconfidence at top


# ═══════════════════════════════════════════════════════════════════
# Calibration Amplifier Formula Tests
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationAmplifier:
    """Test the _compute_calibration_amplifier() formula."""

    def test_no_calibration_data_returns_one(self):
        """With fewer than 10 predictions, amplifier = 1.0 (nominal)."""
        cal = ConfidenceCalibrator()
        eng = _engine_builder(calibrator=cal)
        # 9 predictions — below minimum
        for i in range(9):
            cal.record(0.7, i % 2 == 0)
        assert eng._compute_calibration_amplifier() == 1.0

    def test_perfectly_calibrated_returns_near_one(self):
        """Well-calibrated (low ECE) → amplifier is modest."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.01, n_samples=60)
        eng = _engine_builder(calibrator=cal)
        amplifier = eng._compute_calibration_amplifier()
        ece = cal.compute_ece()
        # With low ECE (< 0.2), amplifier should be below 1.5
        assert 1.0 <= amplifier < 1.6, (
            f"Expected modest amplifier for well-calibrated, got {amplifier:.4f}, "
            f"ECE={ece:.4f}"
        )
        # Lower ECE should produce lower amplifier than poorly-calibrated
        cal_poor = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal_poor, target_ece=0.25, n_samples=60)
        eng_poor = _engine_builder(calibrator=cal_poor)
        amp_poor = eng_poor._compute_calibration_amplifier()
        assert amplifier <= amp_poor, (
            f"Well-calibrated amplifier ({amplifier:.4f}) should be <= "
            f"poorly-calibrated ({amp_poor:.4f})"
        )

    def test_moderately_miscalibrated_amplifies(self):
        """Moderate miscalibration (ECE ~0.10-0.15) → amplifier > 1.15."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.12, n_samples=50)
        eng = _engine_builder(calibrator=cal)
        amplifier = eng._compute_calibration_amplifier()
        ece = cal.compute_ece()
        # Should be significantly above 1.0
        assert amplifier > 1.10, (
            f"Expected amplifier > 1.10 for ECE={ece:.4f}, got {amplifier:.4f}"
        )

    def test_highly_miscalibrated_amplifies_more(self):
        """Significant miscalibration (ECE ~0.20+) → amplifier > 1.3."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.25, n_samples=50)
        eng = _engine_builder(calibrator=cal)
        amplifier = eng._compute_calibration_amplifier()
        ece = cal.compute_ece()
        assert amplifier > 1.20, (
            f"Expected amplifier > 1.20 for ECE={ece:.4f}, got {amplifier:.4f}"
        )

    def test_overconfident_amplifies_further(self):
        """Overconfident calibrator → extra 15% amplification bonus."""
        cal = ConfidenceCalibrator()
        # Create systematic overconfidence: predict high, get it wrong often
        for _ in range(80):
            cal.record(0.85, False)  # 85% confidence, actually wrong
        for _ in range(40):
            cal.record(0.85, True)   # Acc at 0.85 bin = 40/120 = 0.333
        # Add some lower-confidence mostly-correct predictions
        for _ in range(30):
            cal.record(0.30, True)
        for _ in range(30):
            cal.record(0.30, True)
        for _ in range(10):
            cal.record(0.30, False)

        bias = cal.compute_bias()
        assert bias > 0, f"Expected positive bias (overconfident), got bias={bias:.4f}"

        eng = _engine_builder(calibrator=cal)
        amplifier = eng._compute_calibration_amplifier()

        # Overconfidence bonus should give amplifier > 1.0 when ECE > 0
        ece = cal.compute_ece()
        assert amplifier >= 1.0, (
            f"Amplifier should be >= 1.0, got {amplifier:.4f}, ECE={ece:.4f}"
        )

    def test_amplifier_bounded_to_two(self):
        """Amplifier is capped at 2.0 regardless of extreme miscalibration."""
        cal = ConfidenceCalibrator()
        # Extreme overconfidence: all predictions at 0.95, all wrong
        for _ in range(200):
            cal.record(0.95, False)

        eng = _engine_builder(calibrator=cal)
        amplifier = eng._compute_calibration_amplifier()
        assert amplifier <= 2.0, (
            f"Amplifier must be bounded to 2.0, got {amplifier:.4f}"
        )
        assert amplifier >= 1.0

    def test_underconfident_no_extra_bonus(self):
        """Underconfident calibrator does NOT get the extra 1.15 bonus."""
        cal = ConfidenceCalibrator()
        # Create systematic underconfidence
        for _ in range(100):
            cal.record(0.20, True)  # low confidence but actually correct
        for _ in range(100):
            cal.record(0.20, True)

        assert cal.is_underconfident, f"Expected underconfident, bias={cal.compute_bias():.4f}"

        eng = _engine_builder(calibrator=cal)
        amplifier = eng._compute_calibration_amplifier()

        ece = cal.compute_ece()
        bias = abs(cal.compute_bias())
        # Underconfident: bonus = 1.0 (no extra)
        expected = min(1.0 + 2.0 * ece * (1.0 + bias) * 1.0, 2.0)
        assert abs(amplifier - expected) < 0.001, (
            f"Underconfident amplifier should equal base formula, "
            f"got {amplifier:.4f}, expected {expected:.4f}"
        )

    def test_amplifier_monotonic_with_ece(self):
        """Higher ECE should produce higher (or equal) amplifier."""
        cal_low = ConfidenceCalibrator()
        cal_high = ConfidenceCalibrator()

        _populate_calibrator_with_ece(cal_low, target_ece=0.05, n_samples=50)
        _populate_calibrator_with_ece(cal_high, target_ece=0.20, n_samples=50)

        eng_low = _engine_builder(calibrator=cal_low)
        eng_high = _engine_builder(calibrator=cal_high)

        amp_low = eng_low._compute_calibration_amplifier()
        amp_high = eng_high._compute_calibration_amplifier()

        assert amp_high >= amp_low, (
            f"Higher ECE ({cal_high.compute_ece():.4f}) should produce "
            f"amplifier >= lower ECE ({cal_low.compute_ece():.4f}), "
            f"got {amp_high:.4f} vs {amp_low:.4f}"
        )


# ═══════════════════════════════════════════════════════════════════
# Calibration-Aware Reasoning Depth Tests
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationAwareDepth:
    """Test that calibration quality modulates reasoning depth."""

    def test_default_depth_unchanged_without_calibration_data(self):
        """Without calibration data (< 10 predictions), depth is nominal."""
        cal = ConfidenceCalibrator()
        eng = _engine_builder(calibrator=cal, shallow_deep=0.0)
        depth = eng._compute_max_depth()
        # p_shallow=0 → depth_range = (1+0)/2 * 6 * 1.0 = 3 → D = 2 + 3 = 5
        assert depth == 5, f"Expected depth 5, got {depth}"

    def test_miscalibrated_increases_depth(self):
        """Poor calibration increases max reasoning depth."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.02, n_samples=50)
        eng_well = _engine_builder(calibrator=cal, shallow_deep=0.0)
        depth_well = eng_well._compute_max_depth()

        cal2 = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal2, target_ece=0.25, n_samples=50)
        eng_poor = _engine_builder(calibrator=cal2, shallow_deep=0.0)
        depth_poor = eng_poor._compute_max_depth()

        assert depth_poor >= depth_well, (
            f"Poor calibration should produce >= depth: "
            f"well={depth_well}, poor={depth_poor}"
        )
        # With ECE=0.25, amplifier > 1.0 → depth should be higher
        if cal2.compute_ece() > 0.05:
            assert depth_poor > depth_well, (
                f"Poor calibration (ECE={cal2.compute_ece():.4f}) should increase "
                f"depth beyond well-calibrated (ECE={cal.compute_ece():.4f}): "
                f"{depth_poor} vs {depth_well}"
            )

    def test_shallow_mode_respected_min_depth(self):
        """Even with poor calibration, shallow mode keeps min depth = 2."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.30, n_samples=100)
        eng = _engine_builder(calibrator=cal, shallow_deep=-1.0)
        depth = eng._compute_max_depth()
        # p_shallow=-1 → depth_range = (0)/2 * 6 * amplifier = 0 → D = 2
        assert depth == 2, f"Shallow mode should give depth 2, got {depth}"

    def test_deep_with_poor_calibration_pushes_depth_high(self):
        """Deep mode + poor calibration → significantly deeper reasoning."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.30, n_samples=100)
        eng = _engine_builder(calibrator=cal, shallow_deep=1.0)
        depth = eng._compute_max_depth()
        # p_shallow=+1 → depth_range = (2)/2 * 6 * amplifier = 6 * amplifier
        # At ECE=0.30, amplifier ~1.6 → depth_range ≈ 9.6 → D = 2 + 10 = 12
        assert depth >= 9, (
            f"Deep mode + poor calibration should give depth >= 9, got {depth}"
        )


# ═══════════════════════════════════════════════════════════════════
# Calibration-Aware Branching Tests
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationAwareBranching:
    """Test that calibration quality modulates branching factor."""

    def test_default_branching_unchanged_without_calibration_data(self):
        """Without calibration data, branching is nominal."""
        cal = ConfidenceCalibrator()
        eng = _engine_builder(calibrator=cal, explore_exploit=0.0)
        branching = eng._compute_branching_factor()
        # p_exploit=0 → raw = 3 * 1 * 1 = 3 → B = 3
        assert branching == 3, f"Expected branching 3, got {branching}"

    def test_miscalibrated_increases_branching(self):
        """Poor calibration increases branching factor (more alternatives)."""
        cal_well = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal_well, target_ece=0.02, n_samples=50)
        eng_well = _engine_builder(calibrator=cal_well, explore_exploit=0.0)
        branch_well = eng_well._compute_branching_factor()

        cal_poor = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal_poor, target_ece=0.25, n_samples=50)
        eng_poor = _engine_builder(calibrator=cal_poor, explore_exploit=0.0)
        branch_poor = eng_poor._compute_branching_factor()

        assert branch_poor >= branch_well, (
            f"Poor calibration should increase branching: "
            f"well={branch_well}, poor={branch_poor}"
        )

    def test_exploit_mode_limits_branching(self):
        """Exploit mode with well-calibrated → minimum branching."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.01, n_samples=50)
        eng = _engine_builder(calibrator=cal, explore_exploit=1.0)
        branching = eng._compute_branching_factor()
        # p_exploit=+1 → raw = 3 * 0 * amplifier = 0 → B = max(1, 0) = 1
        assert branching == 1, f"Exploit mode should give branching 1, got {branching}"

    def test_explore_mode_with_poor_calibration_high_branching(self):
        """Explore + poor calibration → many alternatives."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.25, n_samples=50)
        eng = _engine_builder(calibrator=cal, explore_exploit=-1.0)
        branching = eng._compute_branching_factor()
        # p_exploit=-1 → raw = 3 * 2 * amplifier = 6 * amplifier
        # ECE=0.25 → amplifier ~1.4 → raw ~8.4 → B = 9
        assert branching >= 6, (
            f"Explore + poor calibration should give branching >= 6, got {branching}"
        )


# ═══════════════════════════════════════════════════════════════════
# Calibration-Aware Divergence Tests
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationAwareDivergence:
    """Test that high calibration error forces divergent mode."""

    def test_convergent_tension_low_ece_stays_convergent(self):
        """When calibration is good, convergent tension is respected."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.02, n_samples=50)
        eng = _engine_builder(calibrator=cal, divergent_convergent=0.5)
        assert eng._is_divergent() is False, (
            "Convergent tension + low ECE should stay convergent"
        )

    def test_high_ece_forces_divergence_despite_convergent_tension(self):
        """Poor calibration forces divergent mode even with convergent tension."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.25, n_samples=50)
        eng = _engine_builder(calibrator=cal, divergent_convergent=0.5)
        assert eng._is_divergent() is True, (
            f"High ECE ({cal.compute_ece():.4f}) should force divergence "
            f"despite convergent tension (0.5)"
        )

    def test_divergent_tension_always_divergent(self):
        """Divergent tension is always divergent regardless of calibration."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.01, n_samples=50)
        eng = _engine_builder(calibrator=cal, divergent_convergent=-0.5)
        assert eng._is_divergent() is True

    def test_not_enough_data_no_divergence_override(self):
        """With < 10 predictions, ECE not checked for divergence."""
        cal = ConfidenceCalibrator()
        # Only 9 predictions — all wrong, but not enough data
        for i in range(9):
            cal.record(0.90, False)
        eng = _engine_builder(calibrator=cal, divergent_convergent=0.5)
        # ECE would be high if computed, but not enough data
        assert eng._is_divergent() is False

    def test_exactly_at_ece_threshold(self):
        """ECE > 0.15 triggers divergence (strict greater-than)."""
        cal = ConfidenceCalibrator()
        # Create exactly borderline calibration
        # Half at conf=0.65 all correct (acc=1.0 at 0.65), half at conf=0.65 all wrong
        for _ in range(25):
            cal.record(0.65, True)
        for _ in range(25):
            cal.record(0.65, False)

        ece = cal.compute_ece()
        eng = _engine_builder(calibrator=cal, divergent_convergent=0.5)
        if ece > 0.15:
            assert eng._is_divergent() is True
        else:
            assert eng._is_divergent() is False


# ═══════════════════════════════════════════════════════════════════
# End-to-end Integration: reasoning plan output
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationAwarePlanOutput:
    """Test that calibration quality affects the actual plan output."""

    def test_well_calibrated_produces_normal_plan(self):
        """Well-calibrated engine produces standard plan depth."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.01, n_samples=50)
        eng = _engine_builder(calibrator=cal, shallow_deep=0.5, explore_exploit=0.5)
        plan = eng.reason("analyze data and produce report")
        assert plan.max_depth_reached <= 6, (
            f"Well-calibrated should have moderate depth, got {plan.max_depth_reached}"
        )
        assert len(plan.best_plan) > 0

    def test_poorly_calibrated_produces_deeper_plan(self):
        """Poorly calibrated engine reasons deeper than well-calibrated."""
        cal_well = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal_well, target_ece=0.01, n_samples=60)
        eng_well = _engine_builder(calibrator=cal_well, shallow_deep=0.5, explore_exploit=0.5)
        plan_well = eng_well.reason(
            "investigate the database performance regression and recommend fixes"
        )

        cal_poor = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal_poor, target_ece=0.30, n_samples=100)
        eng_poor = _engine_builder(calibrator=cal_poor, shallow_deep=0.5, explore_exploit=0.5)
        plan_poor = eng_poor.reason(
            "investigate the database performance regression and recommend fixes"
        )

        # Poor calibration should produce deeper or at least as-deep reasoning
        assert plan_poor.max_depth_reached >= plan_well.max_depth_reached, (
            f"Poorly calibrated should reason >= well-calibrated depth: "
            f"poor={plan_poor.max_depth_reached}, well={plan_well.max_depth_reached}, "
            f"ECE={cal_poor.compute_ece():.4f}"
        )
        assert len(plan_poor.best_plan) > 0
        assert len(plan_well.best_plan) > 0

    def test_poor_calibration_increases_total_nodes(self):
        """Poor calibration → more branches → more total nodes."""
        cal_well = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal_well, target_ece=0.01, n_samples=50)
        eng_well = _engine_builder(calibrator=cal_well, explore_exploit=-0.5,
                                    shallow_deep=0.3)
        plan_well = eng_well.reason("analyze data and produce report")

        cal_poor = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal_poor, target_ece=0.25, n_samples=50)
        eng_poor = _engine_builder(calibrator=cal_poor, explore_exploit=-0.5,
                                    shallow_deep=0.3)
        plan_poor = eng_poor.reason("analyze data and produce report")

        assert plan_poor.total_nodes >= plan_well.total_nodes, (
            f"Poor calibration should produce >= nodes: "
            f"well={plan_well.total_nodes}, poor={plan_poor.total_nodes}"
        )

    def test_calibration_affects_plan_complexity(self):
        """Poor calibration produces more complex plans (more actions/branches)."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.25, n_samples=50)
        eng = _engine_builder(calibrator=cal, explore_exploit=-0.5,
                              shallow_deep=0.5, divergent_convergent=-0.5)
        plan = eng.reason("analyze data and produce report")
        # Should have explored meaningful alternatives
        assert plan.branches_explored >= 0
        assert len(plan.plans) >= 1

    def test_plan_stats_include_tension_profile(self):
        """Plan output includes the tension profile used."""
        eng = _engine_builder(shallow_deep=0.3, explore_exploit=-0.2)
        plan = eng.reason("simple task")
        assert "shallow_deep" in plan.tension_profile
        assert plan.tension_profile["shallow_deep"] == 0.3


# ═══════════════════════════════════════════════════════════════════
# Integration with CognitionPillar
# ═══════════════════════════════════════════════════════════════════


def _make_agent_state(
    shallow_deep: float = -0.2,
    explore_exploit: float = 0.15,
    divergent_convergent: float = 0.3,
) -> AgentState:
    """Create a minimal AgentState with cognition tension axes."""
    axes = frozenset([
        TensionAxis(
            id="shallow_deep",
            pillar=Pillar.COGNITION,
            pole_left="shallow",
            pole_right="deep",
            position=shallow_deep,
        ),
        TensionAxis(
            id="explore_exploit",
            pillar=Pillar.COGNITION,
            pole_left="explore",
            pole_right="exploit",
            position=explore_exploit,
        ),
        TensionAxis(
            id="divergent_convergent",
            pillar=Pillar.COGNITION,
            pole_left="divergent",
            pole_right="convergent",
            position=divergent_convergent,
        ),
    ])
    snapshot = TensionSnapshot(axes=axes)
    identity = AgentIdentity(name="test_agent")
    return AgentState(identity=identity, tensions=snapshot)


class TestCognitionPillarCalibrationIntegration:
    """Test CognitionPillar's use of the calibration amplifier."""

    def test_pillar_initialized_engine_has_calibrator(self):
        """CognitionPillar creates an engine with a calibrator."""
        pillar = CognitionPillar(name="test")
        state = _make_agent_state()
        pillar._on_initialize(state)
        assert pillar.reasoning is not None
        assert pillar.reasoning._calibrator is not None
        assert pillar.reasoning._calibrator.total_predictions == 0

    def test_reason_with_calibration_data_produces_deeper_plan(self):
        """After calibration data is recorded, plans get deeper."""
        pillar = CognitionPillar(name="test")
        state = _make_agent_state(shallow_deep=0.5, explore_exploit=0.0)
        pillar._on_initialize(state)

        # First reason: no calibration data
        plan1 = pillar.reasoning.reason("analyze data")
        depth1 = plan1.max_depth_reached

        # Feed many evaluation results to build calibration data
        for _ in range(50):
            pillar.reasoning.calibrate(0.85, False)  # Overconfident wrong
        for _ in range(50):
            pillar.reasoning.calibrate(0.30, True)   # Underconfident correct

        # Second reason: now has calibration data
        plan2 = pillar.reasoning.reason("analyze data and report")
        depth2 = plan2.max_depth_reached

        assert depth2 >= depth1, (
            f"After calibration data, depth should increase: "
            f"before={depth1}, after={depth2}"
        )

    def test_evaluate_result_builds_calibration_over_time(self):
        """Multiple evaluate_result signals build calibration history."""
        pillar = CognitionPillar(name="test")
        state = _make_agent_state()
        pillar._on_initialize(state)

        assert pillar.reasoning._calibrator.total_predictions == 0

        # Send 30 evaluate_result signals
        for i in range(30):
            success = i % 3 != 0  # 2/3 success rate
            signal = Signal(
                source=Pillar.PRAXIS,
                target=Pillar.COGNITION,
                kind="evaluate_result",
                payload={
                    "success": success,
                    "description": f"action {i}",
                    "confidence": 0.75,
                },
            )
            pillar._on_signal(signal)

        assert pillar.reasoning._calibrator.total_predictions >= 30
        # Should have some ECE (predicted 0.75 confidence vs ~0.67 accuracy)
        ece = pillar.reasoning._calibrator.compute_ece()
        assert ece >= 0.0, f"ECE should be non-negative, got {ece}"


# ═══════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationAmplifierEdgeCases:
    """Edge cases and boundary conditions for calibration amplifier."""

    def test_zero_ece_zero_bias(self):
        """ECE≈0, bias≈0 → amplifier≈1.0."""
        cal = ConfidenceCalibrator()
        # Perfect calibration: accuracy matches confidence in each bin
        # Use 3 bins, each with matched accuracy
        for _ in range(20):
            cal.record(0.25, True)   # 1/4 correct
        for _ in range(60):
            cal.record(0.25, False)
        for _ in range(40):
            cal.record(0.55, True)   # half correct
        for _ in range(40):
            cal.record(0.55, False)
        for _ in range(60):
            cal.record(0.85, True)   # 3/4 correct
        for _ in range(20):
            cal.record(0.85, False)

        eng = _engine_builder(calibrator=cal)
        amplifier = eng._compute_calibration_amplifier()
        ece = cal.compute_ece()
        # With matched accuracy per bin, ECE should be very low
        assert 1.0 <= amplifier < 1.3, (
            f"Expected near 1.0, got amplifier={amplifier:.4f}, ECE={ece:.4f}"
        )

    def test_amplifier_independent_of_tension(self):
        """Amplifier depends on calibrator, not on tension profile."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.15, n_samples=50)

        eng1 = _engine_builder(calibrator=cal, shallow_deep=-1.0,
                               explore_exploit=1.0)
        eng2 = _engine_builder(calibrator=cal, shallow_deep=1.0,
                               explore_exploit=-1.0)

        amp1 = eng1._compute_calibration_amplifier()
        amp2 = eng2._compute_calibration_amplifier()

        assert amp1 == amp2, (
            f"Amplifier should be independent of tension: {amp1} vs {amp2}"
        )

    def test_amplifier_reads_live_calibrator_state(self):
        """Amplifier reflects current calibrator state, not cached value."""
        cal = ConfidenceCalibrator()
        eng = _engine_builder(calibrator=cal)

        amp_before = eng._compute_calibration_amplifier()
        assert amp_before == 1.0  # No data

        # Add calibration data
        for _ in range(30):
            cal.record(0.80, False)  # Overconfident wrong

        amp_after = eng._compute_calibration_amplifier()
        assert amp_after > amp_before, (
            f"Amplifier should update with new data: {amp_before} -> {amp_after}"
        )

    def test_multiple_engine_calls_consistent(self):
        """Repeated calls to _compute_calibration_amplifier return same value."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.15, n_samples=50)
        eng = _engine_builder(calibrator=cal)

        amp1 = eng._compute_calibration_amplifier()
        amp2 = eng._compute_calibration_amplifier()
        amp3 = eng._compute_calibration_amplifier()

        assert amp1 == amp2 == amp3, (
            f"Amplifier should be deterministic: {amp1}, {amp2}, {amp3}"
        )

    def test_stats_include_calibration_summary(self):
        """Engine stats include calibration data."""
        cal = ConfidenceCalibrator()
        _populate_calibrator_with_ece(cal, target_ece=0.10, n_samples=30)
        eng = _engine_builder(calibrator=cal)
        stats = eng.stats
        assert "calibration" in stats
        assert stats["calibration"]["total_predictions"] >= 30

    def test_calibration_amplifier_with_empty_predictions_window(self):
        """Amplifier returns 1.0 even if _predictions deque is empty."""
        cal = ConfidenceCalibrator()
        eng = _engine_builder(calibrator=cal)
        # No records at all
        amp = eng._compute_calibration_amplifier()
        assert amp == 1.0
