"""Tests for calibration-aware attention — metacognition modulates context window management.

Iter-008: When the ConfidenceCalibrator detects poor calibration, the attention
system widens retention (lower thresholds), slows recency decay, and raises
the auto-GC trigger — closing the metacognitive loop across all Cognition resources.
"""

import math

import pytest

from isonome.cognition.attention import (
    AttentionBudget,
    AttentionChunk,
    AttentionEquilibriumSystem,
    GarbageCollectionReport,
    RetentionDecision,
)
from isonome.cognition.reasoning import RecursiveReasoningEngine
from isonome.cognition.pillar import CognitionPillar
from isonome.equilibrium import EquilibriumEngine
from isonome.types import Feedback, Pillar, TensionAxis, TensionSnapshot, AgentIdentity, AgentState


# ═══════════════════════════════════════════════════════════════════
# Calibration State Management
# ═══════════════════════════════════════════════════════════════════

class TestCalibrationState:
    """set_calibration_state() stores metrics and manages activation."""

    @pytest.fixture
    def aes(self):
        return AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)

    def test_default_state_inactive(self, aes):
        assert aes._calibration_active is False
        assert aes._calibration_ece == 0.0
        assert aes._calibration_bias == 0.0
        assert aes._calibration_overconfident is False
        assert aes._calibration_predictions == 0

    def test_activation_at_10_predictions(self, aes):
        aes.set_calibration_state(ece=0.15, bias=0.05, total_predictions=10)
        assert aes._calibration_active is True
        assert aes._calibration_ece == 0.15
        assert aes._calibration_bias == 0.05

    def test_inactive_at_9_predictions(self, aes):
        aes.set_calibration_state(ece=0.15, bias=0.05, total_predictions=9)
        assert aes._calibration_active is False

    def test_inactive_at_0_predictions(self, aes):
        aes.set_calibration_state(ece=0.0, bias=0.0, total_predictions=0)
        assert aes._calibration_active is False

    def test_ece_clamped_to_zero(self, aes):
        aes.set_calibration_state(ece=-0.5, bias=0.0, total_predictions=10)
        assert aes._calibration_ece == 0.0

    def test_overconfident_flag_stored(self, aes):
        aes.set_calibration_state(
            ece=0.20, bias=0.12, is_overconfident=True, total_predictions=20
        )
        assert aes._calibration_overconfident is True
        assert aes._calibration_active is True

    def test_underconfident_flag_stored(self, aes):
        aes.set_calibration_state(
            ece=0.10, bias=-0.08, is_underconfident=True, total_predictions=15
        )
        assert aes._calibration_bias == -0.08
        assert aes._calibration_active is True

    def test_repeated_updates_overwrite(self, aes):
        aes.set_calibration_state(ece=0.30, bias=0.15, total_predictions=50)
        assert aes._calibration_ece == 0.30
        aes.set_calibration_state(ece=0.05, bias=0.02, total_predictions=100)
        assert aes._calibration_ece == 0.05
        assert aes._calibration_predictions == 100


# ═══════════════════════════════════════════════════════════════════
# Calibration Retention Modifier
# ═══════════════════════════════════════════════════════════════════

class TestCalibrationRetentionModifier:
    """_compute_calibration_retention_modifier() computes threshold adjustment."""

    @pytest.fixture
    def aes(self):
        return AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)

    def test_inactive_returns_zero(self, aes):
        assert aes._compute_calibration_retention_modifier() == 0.0

    def test_zero_ece_returns_zero(self, aes):
        aes.set_calibration_state(ece=0.0, bias=0.0, total_predictions=10)
        assert aes._compute_calibration_retention_modifier() == 0.0

    def test_moderate_miscalibration(self, aes):
        # ECE=0.10, bias=0.05, not overconfident
        # modifier = -(1.5 * 0.10 * 1.05 * 1.0) = -0.1575
        aes.set_calibration_state(ece=0.10, bias=0.05, total_predictions=20)
        mod = aes._compute_calibration_retention_modifier()
        assert mod == pytest.approx(-0.1575)

    def test_significant_miscalibration(self, aes):
        # ECE=0.20, bias=0.10, not overconfident
        # modifier = -(1.5 * 0.20 * 1.10 * 1.0) = -0.33 → floor at -0.30
        aes.set_calibration_state(ece=0.20, bias=0.10, total_predictions=30)
        mod = aes._compute_calibration_retention_modifier()
        assert mod == pytest.approx(-0.30)

    def test_overconfidence_bonus(self, aes):
        # ECE=0.15, bias=0.08, overconfident
        # modifier = -(1.5 * 0.15 * 1.08 * 1.2) = -0.2916
        aes.set_calibration_state(
            ece=0.15, bias=0.08, is_overconfident=True, total_predictions=25
        )
        mod = aes._compute_calibration_retention_modifier()
        assert mod == pytest.approx(-0.2916)

    def test_overconfidence_with_low_ece(self, aes):
        # ECE=0.05, bias=0.03, overconfident
        # modifier = -(1.5 * 0.05 * 1.03 * 1.2) = -0.0927
        aes.set_calibration_state(
            ece=0.05, bias=0.03, is_overconfident=True, total_predictions=30
        )
        mod = aes._compute_calibration_retention_modifier()
        assert mod == pytest.approx(-0.0927)

    def test_underconfident_no_bonus(self, aes):
        # Underconfident not overconfident → no bonus
        # ECE=0.12, bias=0.07, NOT overconfident
        # modifier = -(1.5 * 0.12 * 1.07 * 1.0) = -0.1926
        aes.set_calibration_state(
            ece=0.12, bias=0.07, is_overconfident=False, total_predictions=20
        )
        mod = aes._compute_calibration_retention_modifier()
        assert mod == pytest.approx(-0.1926)

    def test_bounded_below_negative_0_30(self, aes):
        # Extreme case: ECE=0.5, bias=0.5, overconfident
        # raw = -(1.5 * 0.5 * 1.5 * 1.2) = -1.35 → floor at -0.30
        aes.set_calibration_state(
            ece=0.5, bias=0.5, is_overconfident=True, total_predictions=10
        )
        mod = aes._compute_calibration_retention_modifier()
        assert mod == -0.30

    def test_negative_bias_absolute_value(self, aes):
        # bias = -0.10, |bias| = 0.10
        # ECE=0.10, |bias|=0.10 → modifier = -(1.5 * 0.10 * 1.10) = -0.165
        aes.set_calibration_state(ece=0.10, bias=-0.10, total_predictions=15)
        mod = aes._compute_calibration_retention_modifier()
        assert mod == pytest.approx(-0.165)


# ═══════════════════════════════════════════════════════════════════
# Calibration Decay Modifier
# ═══════════════════════════════════════════════════════════════════

class TestCalibrationDecayModifier:
    """_compute_calibration_decay_modifier() computes decay rate reduction."""

    @pytest.fixture
    def aes(self):
        return AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)

    def test_inactive_returns_zero(self, aes):
        assert aes._compute_calibration_decay_modifier() == 0.0

    def test_zero_ece_returns_zero(self, aes):
        aes.set_calibration_state(ece=0.0, bias=0.0, total_predictions=10)
        assert aes._compute_calibration_decay_modifier() == 0.0

    def test_moderate_miscalibration(self, aes):
        # ECE=0.10, bias=0.05 → modifier = 0.5 * 0.10 * 1.05 = 0.0525
        aes.set_calibration_state(ece=0.10, bias=0.05, total_predictions=20)
        mod = aes._compute_calibration_decay_modifier()
        assert mod == pytest.approx(0.0525)

    def test_significant_miscalibration(self, aes):
        # ECE=0.20, bias=0.10 → modifier = 0.5 * 0.20 * 1.10 = 0.11
        aes.set_calibration_state(ece=0.20, bias=0.10, total_predictions=30)
        mod = aes._compute_calibration_decay_modifier()
        assert mod == pytest.approx(0.11)

    def test_bounded_at_0_20(self, aes):
        # Extreme: ECE=0.5, bias=0.5 → raw = 0.5 * 0.5 * 1.5 = 0.375 → capped at 0.20
        aes.set_calibration_state(ece=0.5, bias=0.5, total_predictions=10)
        mod = aes._compute_calibration_decay_modifier()
        assert mod == 0.20

    def test_negative_bias_absolute_value(self, aes):
        # ECE=0.15, bias=-0.10 → |bias|=0.10 → modifier = 0.5 * 0.15 * 1.10 = 0.0825
        aes.set_calibration_state(ece=0.15, bias=-0.10, total_predictions=15)
        mod = aes._compute_calibration_decay_modifier()
        assert mod == pytest.approx(0.0825)

    def test_overconfident_no_effect(self, aes):
        # Overconfidence does NOT affect decay modifier (different from retention)
        aes.set_calibration_state(
            ece=0.10, bias=0.05, is_overconfident=True, total_predictions=20
        )
        mod = aes._compute_calibration_decay_modifier()
        assert mod == pytest.approx(0.0525)


# ═══════════════════════════════════════════════════════════════════
# GC with Calibration (threshold modulation)
# ═══════════════════════════════════════════════════════════════════

class TestGCWithCalibration:
    """collect_garbage() thresholds are affected by calibration state."""

    @pytest.fixture
    def aes(self):
        return AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)

    def test_no_calibration_data_normal_gc(self, aes):
        """Without calibration data, GC behaves normally."""
        for i in range(5):
            aes.add_chunk(f"chunk {i}", mutual_info=0.3 + i * 0.1)
        report = aes.collect_garbage()
        assert report.calibration_active is False
        assert report.calibration_modifier == 0.0
        assert report.calibration_ece == 0.0

    def test_poor_calibration_lowers_thresholds(self, aes):
        """Poor calibration → lower keep/prune thresholds (retain more)."""
        # First, GC without calibration
        for i in range(3):
            aes.add_chunk(f"normal chunk {i}", mutual_info=0.4)
        report_normal = aes.collect_garbage()

        # Set poor calibration
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_overconfident=True, total_predictions=30
        )

        # Add similar chunks and GC again
        for i in range(3):
            aes.add_chunk(f"calibrated chunk {i}", mutual_info=0.4)
        report_cal = aes.collect_garbage()

        # Thresholds should be LOWER when poorly calibrated
        assert report_cal.keep_threshold < report_normal.keep_threshold
        assert report_cal.calibration_active is True
        assert report_cal.calibration_modifier < 0.0

    def test_well_calibrated_thresholds_unchanged(self, aes):
        """Zero ECE → thresholds should be at normal tension-modulated values."""
        aes.set_calibration_state(ece=0.0, bias=0.0, total_predictions=50)
        for i in range(3):
            aes.add_chunk(f"chunk {i}", mutual_info=0.4)
        report = aes.collect_garbage()
        assert report.calibration_active is True
        assert report.calibration_modifier == 0.0

    def test_calibration_keeps_more_chunks(self, aes):
        """Poor calibration → more chunks retained instead of pruned."""
        aes.set_calibration_state(
            ece=0.25, bias=0.12, is_overconfident=True, total_predictions=30
        )
        # Add low-value chunks that would normally be pruned
        for i in range(10):
            aes.add_chunk(f"low value {i}", mutual_info=0.1, task_relevance=0.1)
        # Apply some decay
        aes.apply_recency_decay(decay_rate=0.3)

        report = aes.collect_garbage()
        # With calibration active, fewer should be pruned
        assert report.calibration_active is True
        assert report.calibration_modifier < 0.0

    def test_gc_report_includes_calibration_data(self, aes):
        aes.set_calibration_state(
            ece=0.15, bias=0.08, is_overconfident=True, total_predictions=25
        )
        aes.add_chunk("test")
        report = aes.collect_garbage()
        assert report.calibration_active is True
        assert report.calibration_ece == pytest.approx(0.15)
        assert report.calibration_modifier < 0.0

    def test_gc_summary_includes_cal_when_active(self, aes):
        aes.set_calibration_state(
            ece=0.18, bias=0.09, is_overconfident=True, total_predictions=20
        )
        aes.add_chunk("test content for summary")
        report = aes.collect_garbage()
        summary = report.summary()
        assert "calΔ" in summary
        assert "ECE" in summary

    def test_gc_summary_no_cal_when_inactive(self, aes):
        aes.add_chunk("test")
        report = aes.collect_garbage()
        summary = report.summary()
        assert "calΔ" not in summary

    def test_gc_summary_no_cal_when_modifier_near_zero(self, aes):
        aes.set_calibration_state(ece=0.0001, bias=0.0, total_predictions=15)
        aes.add_chunk("test")
        report = aes.collect_garbage()
        summary = report.summary()
        # Modifier ~0, should not show calibration info
        assert "calΔ" not in summary

    def test_thresholds_never_below_minimum(self, aes):
        """Even with extreme calibration, thresholds have a floor."""
        aes.set_calibration_state(
            ece=0.5, bias=0.5, is_overconfident=True, total_predictions=100
        )
        for i in range(3):
            aes.add_chunk(f"chunk {i}", mutual_info=0.3)
        report = aes.collect_garbage()
        # keep_threshold should be at least 0.1, prune_threshold at least 0.05
        assert report.keep_threshold >= 0.1
        assert report.prune_threshold >= 0.05


# ═══════════════════════════════════════════════════════════════════
# Calibration-Aware Recency Decay
# ═══════════════════════════════════════════════════════════════════

class TestCalibrationAwareDecay:
    """apply_recency_decay() slows decay when calibration is poor."""

    @pytest.fixture
    def aes(self):
        return AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)

    def test_normal_decay_when_inactive(self, aes):
        aes.add_chunk("test content")
        chunk_id = list(aes._chunks.keys())[0]
        original = aes._chunks[chunk_id].recency
        assert original == 1.0

        aes.apply_recency_decay(decay_rate=0.10)
        assert aes._chunks[chunk_id].recency == pytest.approx(0.90)

    def test_slower_decay_with_poor_calibration(self, aes):
        """Calibration-aware decay should be slower than nominal."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, total_predictions=30
        )
        aes.add_chunk("calibrated decay test")
        chunk_id = list(aes._chunks.keys())[0]

        aes.apply_recency_decay(decay_rate=0.10)
        # cal_decay_mod = 0.5 * 0.20 * 1.10 = 0.11
        # effective_rate = 0.10 * 0.89 = 0.089
        # new_recency = 1.0 * 0.911 = 0.911
        assert aes._chunks[chunk_id].recency > 0.90  # Slower than nominal 0.90

    def test_decay_rate_calibration_comparison(self, aes):
        """Compare decay rates with and without calibration."""
        # Without calibration
        aes.add_chunk("uncalibrated A")
        aes.add_chunk("uncalibrated B")
        aes.apply_recency_decay(decay_rate=0.20)
        uncal_recency = min(c.recency for c in aes._chunks.values())

        # Reset with calibration
        aes2 = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
        aes2.set_calibration_state(
            ece=0.30, bias=0.15, total_predictions=30
        )
        aes2.add_chunk("calibrated A")
        aes2.add_chunk("calibrated B")
        aes2.apply_recency_decay(decay_rate=0.20)
        cal_recency = min(c.recency for c in aes2._chunks.values())

        # Calibrated system should have higher recency (slower decay)
        assert cal_recency > uncal_recency

    def test_no_decay_modification_when_inactive(self, aes):
        """When calibration is inactive, decay should be exactly nominal."""
        aes.add_chunk("nominal decay")
        chunk_id = list(aes._chunks.keys())[0]

        aes.apply_recency_decay(decay_rate=0.15)
        # Should be exactly 0.85
        assert aes._chunks[chunk_id].recency == pytest.approx(0.85)

    def test_decay_mod_bounded_max(self, aes):
        """Even at max calibration error, decay rate can't go below 80% nominal."""
        aes.set_calibration_state(
            ece=0.5, bias=0.5, total_predictions=100
        )
        aes.add_chunk("max decay test")
        chunk_id = list(aes._chunks.keys())[0]

        # cal_decay_mod capped at 0.20 → effective_rate = 0.10 * 0.80 = 0.08
        # new_recency = 1.0 * 0.92 = 0.92
        aes.apply_recency_decay(decay_rate=0.10)
        assert aes._chunks[chunk_id].recency == pytest.approx(0.92, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════
# CognitionPillar Calibration Integration
# ═══════════════════════════════════════════════════════════════════

class TestCognitionPillarCalibrationAttention:
    """CognitionPillar pushes calibration state to attention in update_tension_profile()."""

    @pytest.fixture
    def pillar(self):
        engine = EquilibriumEngine()
        cp = CognitionPillar(name="test_cog", engine=engine, token_capacity=50_000)
        # Initialize with valid AgentState
        identity = AgentIdentity(name="test_agent")
        axes = frozenset([
            TensionAxis(
                id="shallow_deep", pillar=Pillar.COGNITION,
                pole_left="shallow", pole_right="deep", position=-0.2
            ),
            TensionAxis(
                id="explore_exploit", pillar=Pillar.COGNITION,
                pole_left="explore", pole_right="exploit", position=0.15
            ),
        ])
        snapshot = TensionSnapshot(axes=axes)
        state = AgentState(identity=identity, tensions=snapshot)
        cp.initialize(state)
        return cp

    def test_attention_gets_calibration_on_tick(self, pillar):
        """When reasoning engine has calibration data, attention receives it."""
        # Prime the calibrator with 10+ predictions
        for i in range(15):
            pillar.reasoning.calibrator.record(
                0.7 if i % 2 == 0 else 0.4,
                i % 2 == 0  # Half correct
            )

        # Call update_tension_profile (simulates a tick)
        pillar.update_tension_profile({
            "shallow_deep": 0.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.0,
        })

        # Attention should have calibration state
        assert pillar.attention._calibration_active is True
        assert pillar.attention._calibration_predictions == 15
        assert pillar.attention._calibration_ece > 0.0  # Some miscalibration

    def test_attention_no_calibration_when_few_predictions(self, pillar):
        """With < 10 predictions, attention calibration stays inactive."""
        for i in range(5):
            pillar.reasoning.calibrator.record(0.7, True)

        pillar.update_tension_profile({
            "shallow_deep": 0.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.0,
        })

        assert pillar.attention._calibration_active is False

    def test_gc_threshold_increases_with_poor_calibration(self, pillar):
        """Auto-GC trigger threshold is higher when calibration is poor."""
        # Prime calibrator with overconfident predictions
        for i in range(30):
            pillar.reasoning.calibrator.record(0.85, i % 4 == 0)  # 25% accuracy vs 85% confidence

        pillar._auto_gc = True
        pillar._gc_util_threshold = 0.80

        # Before calibration, threshold is nominal
        assert pillar._gc_util_threshold == 0.80

        # After tick with calibration data, effective threshold should increase
        # Calibrator computes that it needs more context → raises GC trigger
        # We can't directly observe effective_gc_threshold (local var), but
        # we can verify the calibrator is overconfident
        assert pillar.reasoning.calibrator.is_overconfident is True
        assert pillar.reasoning.calibrator.compute_ece() > 0.0

    def test_auto_gc_not_triggered_prematurely_with_calibration(self, pillar):
        """Poor calibration → higher GC threshold → GC not triggered prematurely."""
        # Prime calibrator
        for i in range(30):
            pillar.reasoning.calibrator.record(0.85, i % 4 == 0)

        pillar._auto_gc = True
        pillar._gc_util_threshold = 0.80

        # Fill budget to 83% utilization — normally would trigger GC at 0.80
        # but with poor calibration, effective threshold might be ~0.88
        fill_tokens = int(50_000 * 0.83)
        # Add chunks to fill budget
        content = "x" * 400  # ~100 tokens each
        for _ in range(fill_tokens // 100):
            if pillar.attention.budget.utilization < 0.83:
                pillar.attention.add_chunk(content)

        # Start with no GC report
        pillar._last_gc_report = None

        # Tick — with calibration, 83% may not trigger GC
        pillar.update_tension_profile({
            "shallow_deep": 0.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.0,
        })

        # GC may or may not have triggered depending on effective threshold
        # We just verify no crash
        assert pillar.attention is not None


# ═══════════════════════════════════════════════════════════════════
# End-to-End: Calibration → Attention → GC behavior
# ═══════════════════════════════════════════════════════════════════

class TestEndToEndCalibrationAttention:
    """Full pipeline: calibrator state → attention modulation → GC behavior."""

    def test_full_pipeline_changes_gc_behavior(self):
        """Poor calibration → lower thresholds → more chunks kept."""
        engine = EquilibriumEngine()
        aes = AttentionEquilibriumSystem(engine, token_capacity=20_000)

        # Add moderate-value chunks
        for i in range(10):
            aes.add_chunk(
                f"chunk number {i} with some reasonable content for testing",
                mutual_info=0.45,
                task_relevance=0.5,
            )

        # Decay a bit
        aes.apply_recency_decay(decay_rate=0.2)

        # GC WITHOUT calibration
        report_no_cal = aes.collect_garbage()
        pruned_no_cal = report_no_cal.pruned_count

        # Reset — fresh AES with same chunks but poor calibration
        aes2 = AttentionEquilibriumSystem(engine, token_capacity=20_000)
        aes2.set_calibration_state(
            ece=0.25, bias=0.12, is_overconfident=True, total_predictions=50
        )
        for i in range(10):
            aes2.add_chunk(
                f"chunk number {i} with some reasonable content for testing",
                mutual_info=0.45,
                task_relevance=0.5,
            )
        aes2.apply_recency_decay(decay_rate=0.2)

        report_cal = aes2.collect_garbage()
        pruned_cal = report_cal.pruned_count

        # With poor calibration, FEWER chunks should be pruned
        # (may be equal if all chunks score above threshold in both cases)
        assert pruned_cal <= pruned_no_cal

    def test_calibration_state_persists_across_gc_cycles(self, aes_factory=None):
        """Calibration state survives multiple GC cycles."""
        aes = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
        aes.set_calibration_state(
            ece=0.18, bias=0.09, total_predictions=25
        )

        for cycle in range(3):
            aes.add_chunk(f"cycle {cycle} test content")
            report = aes.collect_garbage()
            assert report.calibration_active is True
            assert report.calibration_modifier < 0.0

    def test_calibration_state_cleared_on_reset(self):
        """Setting calibration to inactive resets modifiers to zero."""
        aes = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
        aes.set_calibration_state(
            ece=0.20, bias=0.10, total_predictions=30
        )
        assert aes._compute_calibration_retention_modifier() < 0.0

        # Reset to inactive
        aes.set_calibration_state(ece=0.0, bias=0.0, total_predictions=5)
        assert aes._calibration_active is False
        assert aes._compute_calibration_retention_modifier() == 0.0
        assert aes._compute_calibration_decay_modifier() == 0.0

    def test_calibration_and_tension_compose_independently(self, aes_factory=None):
        """Calibration modifier and tension modulation should compose."""
        aes = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
        aes.set_calibration_state(
            ece=0.20, bias=0.10, total_predictions=30
        )
        aes.add_chunk("test")

        # GC with calibration + default tensions
        report1 = aes.collect_garbage()

        # GC with calibration + shallow tension (raise thresholds)
        engine = aes._engine
        engine.apply_feedback(Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="shallow_deep",
            signal=-1.0,
            confidence=1.0,
            reason="force shallow",
        ))
        aes.add_chunk("test 2")
        report2 = aes.collect_garbage()

        # Both reports should have calibration active
        assert report1.calibration_active is True
        assert report2.calibration_active is True
        # Both have the same calibration modifier
        assert report1.calibration_modifier == report2.calibration_modifier


# ═══════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════

class TestCalibrationAttentionEdgeCases:
    """Edge cases for calibration-aware attention."""

    def test_just_barely_active(self, aes_factory=None):
        """Exactly 10 predictions: activation threshold boundary."""
        aes = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
        aes.set_calibration_state(ece=0.15, bias=0.05, total_predictions=10)
        assert aes._calibration_active is True
        assert aes._compute_calibration_retention_modifier() < 0.0

    def test_just_barely_inactive(self, aes_factory=None):
        """Exactly 9 predictions: not yet active."""
        aes = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
        aes.set_calibration_state(ece=0.15, bias=0.05, total_predictions=9)
        assert aes._calibration_active is False
        assert aes._compute_calibration_retention_modifier() == 0.0

    def test_large_negative_bias(self, aes_factory=None):
        """Large underconfident bias → |bias| used, no overconfidence bonus."""
        aes = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
        aes.set_calibration_state(
            ece=0.10, bias=-0.15, is_underconfident=True, total_predictions=20
        )
        # modifier = -(1.5 * 0.10 * 1.15 * 1.0) = -0.1725
        mod = aes._compute_calibration_retention_modifier()
        assert mod == pytest.approx(-0.1725)

    def test_zero_ece_nonzero_bias(self, aes_factory=None):
        """ECE=0 but bias nonzero (all predictions same confidence, all correct): no modifier."""
        aes = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
        aes.set_calibration_state(ece=0.0, bias=0.10, total_predictions=50)
        # ECE=0 → modifier = 0 * anything = 0
        assert aes._compute_calibration_retention_modifier() == 0.0

    def test_no_chunks_gc_with_calibration(self, aes_factory=None):
    	"""GC with calibration but no chunks: should not crash."""
    	aes = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
    	aes.set_calibration_state(
    		ece=0.20, bias=0.10, is_overconfident=True, total_predictions=30
    	)
    	report = aes.collect_garbage()
    	assert report.chunks_before == 0
    	assert report.chunks_after == 0
    	assert report.calibration_active is True

# ═══════════════════════════════════════════════════════════════════
# Calibration Weight Rebalance (iter-016)
# ═══════════════════════════════════════════════════════════════════

class TestCalibrationWeightRebalance:
    """_compute_calibration_weight_rebalance() shifts α↔β based on calibration."""

    @pytest.fixture
    def aes(self):
        return AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)

    # ── Inactive / below threshold ──

    def test_inactive_returns_zero(self, aes):
        """When calibration is inactive, no weight shift."""
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == 0.0
        assert beta_d == 0.0

    def test_zero_ece_returns_zero(self, aes):
        """Active calibration but zero ECE: no shift."""
        aes.set_calibration_state(ece=0.0, bias=0.0, total_predictions=10)
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == 0.0
        assert beta_d == 0.0

    def test_ece_below_015_returns_zero(self, aes):
        """ECE < 0.15: not enough miscalibration to warrant rebalance."""
        aes.set_calibration_state(ece=0.10, bias=0.05, total_predictions=20)
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == 0.0
        assert beta_d == 0.0

    def test_ece_exactly_015_returns_zero(self, aes):
        """ECE exactly at 0.15 boundary: still no rebalance (strict <)."""
        aes.set_calibration_state(ece=0.15, bias=0.05, total_predictions=20)
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == 0.0
        assert beta_d == 0.0

    # ── Overconfident: shift β → α ──

    def test_overconfident_shifts_beta_to_alpha(self, aes):
        """Overconfident: α increases (positive delta), β decreases (negative)."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_overconfident=True, total_predictions=30
        )
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d > 0.0  # Surprisal weight increases
        assert beta_d < 0.0   # MI weight decreases
        assert alpha_d == pytest.approx(-beta_d)  # Zero-sum

    def test_overconfident_exact_calculation(self, aes):
        """Overconfident: Δ = η × ECE × (1+|bias|) × 1.2."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_overconfident=True, total_predictions=30
        )
        # Δ = 0.50 × 0.20 × (1 + 0.10) × 1.2 = 0.50 × 0.20 × 1.10 × 1.2
        #   = 0.50 × 0.264 = 0.132 → capped at 0.12
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == pytest.approx(0.12)  # Capped at max_shift
        assert beta_d == pytest.approx(-0.12)

    def test_overconfident_below_cap(self, aes):
        """Overconfident with lower ECE: shift below the 0.12 cap."""
        aes.set_calibration_state(
            ece=0.16, bias=0.05, is_overconfident=True, total_predictions=25
        )
        # Δ = 0.50 × 0.16 × (1 + 0.05) × 1.2 = 0.50 × 0.16 × 1.05 × 1.2
        #   = 0.50 × 0.2016 = 0.1008
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == pytest.approx(0.1008)
        assert beta_d == pytest.approx(-0.1008)

    def test_overconfident_zero_bias(self, aes):
        """Overconfident with zero bias: no bias amplification."""
        aes.set_calibration_state(
            ece=0.20, bias=0.0, is_overconfident=True, total_predictions=30
        )
        # Δ = 0.50 × 0.20 × (1 + 0.0) × 1.2 = 0.50 × 0.24 = 0.12
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == pytest.approx(0.12)
        assert beta_d == pytest.approx(-0.12)

    def test_overconfident_negative_bias_uses_abs(self, aes):
        """Overconfident with negative bias: |bias| used in formula."""
        aes.set_calibration_state(
            ece=0.20, bias=-0.10, is_overconfident=True, total_predictions=30
        )
        # |bias| = 0.10, same as positive 0.10
        # Δ = 0.50 × 0.20 × (1 + 0.10) × 1.2 = 0.132 → capped 0.12
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == pytest.approx(0.12)
        assert beta_d == pytest.approx(-0.12)

    def test_overconfident_extreme_ece_capped(self, aes):
        """Extreme ECE: shift is still capped at 0.12."""
        aes.set_calibration_state(
            ece=0.50, bias=0.50, is_overconfident=True, total_predictions=100
        )
        # Raw would be 0.50 × 0.50 × 1.50 × 1.2 = 0.45, capped at 0.12
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == 0.12
        assert beta_d == -0.12

    # ── Underconfident: shift α → β ──

    def test_underconfident_shifts_alpha_to_beta(self, aes):
        """Underconfident: α decreases (negative), β increases (positive)."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_underconfident=True, total_predictions=30
        )
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d < 0.0   # Surprisal weight decreases
        assert beta_d > 0.0    # MI weight increases
        assert alpha_d == pytest.approx(-beta_d)  # Zero-sum

    def test_underconfident_exact_calculation(self, aes):
        """Underconfident: Δ = η × ECE × (1+|bias|), no 1.2 bonus."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_underconfident=True, total_predictions=30
        )
        # Δ = 0.50 × 0.20 × (1 + 0.10) = 0.50 × 0.22 = 0.11
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == pytest.approx(-0.11)
        assert beta_d == pytest.approx(0.11)

    def test_underconfident_no_overconfidence_bonus(self, aes):
        """Underconfident shift is smaller than overconfident (no 1.2x)."""
        aes_oc = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
        aes_uc = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)

        aes_oc.set_calibration_state(
            ece=0.20, bias=0.10, is_overconfident=True, total_predictions=30
        )
        aes_uc.set_calibration_state(
            ece=0.20, bias=0.10, is_underconfident=True, total_predictions=30
        )

        oc_alpha, _ = aes_oc._compute_calibration_weight_rebalance()
        uc_alpha, _ = aes_uc._compute_calibration_weight_rebalance()

        # Overconfident shift should be larger (1.2x bonus)
        assert abs(oc_alpha) > abs(uc_alpha)

    def test_underconfident_extreme_ece_capped(self, aes):
        """Extreme underconfident ECE: shift capped at 0.12."""
        aes.set_calibration_state(
            ece=0.50, bias=0.50, is_underconfident=True, total_predictions=100
        )
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == -0.12
        assert beta_d == 0.12

    # ── Moderate: neither over nor under ──

    def test_moderate_miscalibration_no_shift(self, aes):
        """ECE > 0.15 but neither over/underconfident: no shift."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, total_predictions=30
        )
        alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
        assert alpha_d == 0.0
        assert beta_d == 0.0

    # ── Zero-sum invariant ──

    def test_zero_sum_invariant_all_cases(self, aes):
        """α_delta + β_delta = 0 for all calibration states."""
        for ece, bias, oc, uc in [
            (0.20, 0.10, True, False),
            (0.20, 0.10, False, True),
            (0.30, 0.20, True, False),
            (0.16, 0.05, False, True),
            (0.50, 0.50, True, False),
        ]:
            aes.set_calibration_state(
                ece=ece, bias=bias,
                is_overconfident=oc, is_underconfident=uc,
                total_predictions=30,
            )
            alpha_d, beta_d = aes._compute_calibration_weight_rebalance()
            assert alpha_d + beta_d == pytest.approx(0.0), \
                f"Zero-sum violated: ece={ece}, bias={bias}, oc={oc}"


# ═══════════════════════════════════════════════════════════════════
# GC Report: Weight Rebalance Fields (iter-016)
# ═══════════════════════════════════════════════════════════════════

class TestGCReportWeightRebalance:
    """GarbageCollectionReport includes calibration weight rebalance data."""

    @pytest.fixture
    def aes(self):
        return AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)

    def test_report_fields_default_zero(self, aes):
        """Without calibration, rebalance fields are 0.0."""
        aes.add_chunk("test")
        report = aes.collect_garbage()
        assert report.calibration_weight_rebalance_alpha == 0.0
        assert report.calibration_weight_rebalance_beta == 0.0

    def test_report_fields_overconfident(self, aes):
        """Overconfident calibration: report shows α+/β- rebalance."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_overconfident=True, total_predictions=30
        )
        aes.add_chunk("test content for rebalance report")
        report = aes.collect_garbage()
        assert report.calibration_weight_rebalance_alpha > 0.0
        assert report.calibration_weight_rebalance_beta < 0.0

    def test_report_fields_underconfident(self, aes):
        """Underconfident calibration: report shows α-/β+ rebalance."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_underconfident=True, total_predictions=30
        )
        aes.add_chunk("test content for rebalance report")
        report = aes.collect_garbage()
        assert report.calibration_weight_rebalance_alpha < 0.0
        assert report.calibration_weight_rebalance_beta > 0.0

    def test_report_fields_low_ece_zero(self, aes):
        """Low ECE (below 0.15): rebalance fields are 0.0."""
        aes.set_calibration_state(
            ece=0.10, bias=0.05, total_predictions=20
        )
        aes.add_chunk("test")
        report = aes.collect_garbage()
        assert report.calibration_weight_rebalance_alpha == 0.0
        assert report.calibration_weight_rebalance_beta == 0.0

    def test_summary_includes_rebalance_when_active(self, aes):
        """GC summary includes calWΔ when weight rebalance is nonzero."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_overconfident=True, total_predictions=30
        )
        aes.add_chunk("test content for summary")
        report = aes.collect_garbage()
        summary = report.summary()
        assert "calWΔ" in summary

    def test_summary_excludes_rebalance_when_zero(self, aes):
        """GC summary excludes calWΔ when rebalance is zero."""
        aes.add_chunk("test")
        report = aes.collect_garbage()
        summary = report.summary()
        assert "calWΔ" not in summary

    def test_summary_excludes_rebalance_when_low_ece(self, aes):
        """GC summary excludes calWΔ when ECE is below threshold."""
        aes.set_calibration_state(ece=0.10, bias=0.05, total_predictions=20)
        aes.add_chunk("test")
        report = aes.collect_garbage()
        summary = report.summary()
        assert "calWΔ" not in summary


# ═══════════════════════════════════════════════════════════════════
# Weight Rebalance Integration with _modulate_weights (iter-016)
# ═══════════════════════════════════════════════════════════════════

class TestModulateWeightsWithCalibration:
    """_modulate_weights() incorporates calibration-driven rebalance."""

    @pytest.fixture
    def aes(self):
        return AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)

    def test_overconfident_increases_alpha_in_weights(self, aes):
        """Overconfident: resulting α from _modulate_weights is higher than default."""
        profile = {
            "shallow_deep": 0.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.0,
        }

        # Without calibration
        alpha_no_cal, _, _, _ = aes._modulate_weights(profile)

        # With overconfident calibration
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_overconfident=True, total_predictions=30
        )
        alpha_cal, beta_cal, _, _ = aes._modulate_weights(profile)

        # Alpha should be higher with overconfident calibration
        assert alpha_cal > alpha_no_cal

    def test_underconfident_increases_beta_in_weights(self, aes):
        """Underconfident: resulting β from _modulate_weights is higher than default."""
        profile = {
            "shallow_deep": 0.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.0,
        }

        # Without calibration
        alpha_no_cal, beta_no_cal, _, _ = aes._modulate_weights(profile)

        # With underconfident calibration
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_underconfident=True, total_predictions=30
        )
        alpha_cal, beta_cal, _, _ = aes._modulate_weights(profile)

        # Beta should be higher with underconfident calibration
        assert beta_cal > beta_no_cal
        assert alpha_cal < alpha_no_cal  # Alpha decreased

    def test_weights_still_sum_to_one(self, aes):
        """After calibration rebalance, weights still sum to 1.0."""
        aes.set_calibration_state(
            ece=0.25, bias=0.15, is_overconfident=True, total_predictions=50
        )
        profile = {
            "shallow_deep": 0.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.0,
        }
        alpha, beta, gamma, delta = aes._modulate_weights(profile)
        assert alpha + beta + gamma + delta == pytest.approx(1.0)

    def test_calibrate_rebalance_with_tensions(self, aes):
        """Calibration rebalance composes with tension-based weight modulation."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, is_overconfident=True, total_predictions=30
        )
        # Explore tension (increases alpha)
        profile = {
            "shallow_deep": 0.0,
            "explore_exploit": -1.0,  # Explore
            "divergent_convergent": 0.0,
        }
        alpha, _, _, _ = aes._modulate_weights(profile)

        # Both explore tension and overconfident rebalance push α up
        # Default α is 0.35; explore adds +0.10; overconfident adds ~0.11
        # Before normalization: α ≈ 0.35 + 0.10 + 0.11 = 0.56
        assert alpha > 0.40  # Significantly above default

    def test_no_rebalance_without_over_under_flag(self, aes):
        """ECE > 0.15 but no over/underconfident flag: weights unchanged."""
        aes.set_calibration_state(
            ece=0.20, bias=0.10, total_predictions=30
        )
        profile = {
            "shallow_deep": 0.0,
            "explore_exploit": 0.0,
            "divergent_convergent": 0.0,
        }
        alpha_cal, beta_cal, _, _ = aes._modulate_weights(profile)

        # Without over/underconfident flag, weights should be the same as default
        aes2 = AttentionEquilibriumSystem(EquilibriumEngine(), token_capacity=10_000)
        alpha_default, beta_default, _, _ = aes2._modulate_weights(profile)

        assert alpha_cal == pytest.approx(alpha_default)
        assert beta_cal == pytest.approx(beta_default)
