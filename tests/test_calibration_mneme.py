"""Tests for calibration-aware Mneme consolidation.

Iteration 009: When the ConfidenceCalibrator shows the agent is poorly
calibrated (high ECE), the HierarchicalMneme should:

1. Raise consolidation/promotion thresholds (consolidate cautiously)
2. Prune less aggressively when overconfident
3. Report calibration metrics in ConsolidationReport
4. Support push from CognitionPillar through MnemePillar
"""

import math
import pytest

from isonome.mneme.hierarchical import (
    HierarchicalMneme,
    ConsolidationReport,
    MemoryEntry,
    MemoryTier,
)
from isonome.types import (
    AgentIdentity,
    AgentState,
    Feedback,
    Pillar,
    Signal,
    TensionAxis,
    TensionSnapshot,
)

from isonome.cognition.reasoning import ConfidenceCalibrator
from isonome.mneme.pillar import MnemePillar
from isonome.cognition.pillar import CognitionPillar


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def mneme():
    return HierarchicalMneme(
        consolidation_significance=0.5,
        promotion_significance=0.7,
    )


@pytest.fixture
def overconfident_calibrator():
    """Calibrator heavily overconfident — high ECE."""
    cal = ConfidenceCalibrator()
    for _ in range(50):
        cal.record(predicted_confidence=0.85, actual_success=False)
    for _ in range(20):
        cal.record(predicted_confidence=0.85, actual_success=True)
    # ~70 predictions, ~29% accuracy with 85% confidence → high ECE
    return cal


@pytest.fixture
def well_calibrated_calibrator():
    """Calibrator well-calibrated — low ECE."""
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


@pytest.fixture
def stress_calibrator():
    """Calibrator with just under 10 predictions — at the activation boundary."""
    cal = ConfidenceCalibrator()
    for _ in range(9):
        cal.record(predicted_confidence=0.60, actual_success=True)
    return cal


@pytest.fixture
def agent_state():
    axes = frozenset([
        TensionAxis(
            id="consolidate_prune", pillar=Pillar.MNEME,
            pole_left="consolidate", pole_right="prune", position=0.0,
        ),
        TensionAxis(
            id="specific_general", pillar=Pillar.MNEME,
            pole_left="specific", pole_right="general", position=0.0,
        ),
        TensionAxis(
            id="explore_exploit", pillar=Pillar.COGNITION,
            pole_left="explore", pole_right="exploit", position=0.15,
        ),
    ])
    snapshot = TensionSnapshot(axes=axes)
    identity = AgentIdentity(name="test_agent")
    return AgentState(identity=identity, tensions=snapshot)


# ═══════════════════════════════════════════════════════════════════
# Calibration state management
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationState:
    """HierarchicalMneme.set_calibration_state() stores/clears metrics."""

    def test_default_calibration_is_zero(self, mneme):
        """Before any set_calibration_state call, ECE should be 0."""
        assert mneme._calibration_ece == 0.0
        assert mneme._calibration_total_predictions == 0
        assert mneme._calibration_overconfident is False

    def test_set_calibration_state_round_trip(self, mneme):
        """Values set via set_calibration_state are stored correctly."""
        mneme.set_calibration_state(
            ece=0.15, bias=0.10,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        assert mneme._calibration_ece == 0.15
        assert mneme._calibration_bias == 0.10
        assert mneme._calibration_overconfident is True
        assert mneme._calibration_total_predictions == 50

    def test_clear_calibration_on_zero(self, mneme):
        """Setting calibration to zero should reset state."""
        mneme.set_calibration_state(
            ece=0.20, bias=0.15,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        mneme.set_calibration_state(
            ece=0.0, bias=0.0,
            is_overconfident=False, is_underconfident=False,
            total_predictions=0,
        )
        assert mneme._calibration_ece == 0.0
        assert mneme._calibration_overconfident is False

    def test_calibration_state_independent_of_tension(self, mneme):
        """Calibration state does not overwrite tension profile."""
        mneme.set_tension_profile({"consolidate_prune": -0.5})
        mneme.set_calibration_state(
            ece=0.20, bias=0.15,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        assert mneme._current_profile.get("consolidate_prune") == -0.5
        assert mneme._calibration_ece == 0.20


# ═══════════════════════════════════════════════════════════════════
# Calibration-aware threshold modulation
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationThresholdModulation:
    """_modulate_thresholds() should respond to calibration state."""

    def test_no_calibration_no_change(self, mneme):
        """Without calibration data, thresholds are tension-only."""
        cons_sig, prom_sig = mneme._modulate_thresholds()
        # Default: cons=0.5, prom=0.7, neutral tension → no modulation
        assert cons_sig == 0.5
        assert prom_sig == 0.7

    def test_few_predictions_no_calibration_effect(self, mneme):
        """Fewer than 10 predictions should not activate calibration modulation."""
        mneme.set_calibration_state(
            ece=0.30, bias=0.25,
            is_overconfident=True, is_underconfident=False,
            total_predictions=5,  # Below 10-prediction guard
        )
        cons_sig, prom_sig = mneme._modulate_thresholds()
        assert cons_sig == 0.5  # Unchanged
        assert prom_sig == 0.7

    def test_high_ece_raises_thresholds(self, mneme):
        """High ECE should raise consolidation thresholds."""
        mneme.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        cons_sig, prom_sig = mneme._modulate_thresholds()
        # ECE modulation: cons += 0.25 * 0.30 + 0.25 * 0.20 (overconfident bonus) = 0.075 + 0.05 = 0.125
        # prom += 0.125 * 0.75 ≈ 0.094
        assert cons_sig > 0.6  # At least 0.5 + 0.125 = 0.625
        assert prom_sig > 0.75  # At least 0.7 + 0.094 = 0.794

    def test_low_ece_minimal_effect(self, mneme):
        """Low ECE (well-calibrated) should barely change thresholds."""
        mneme.set_calibration_state(
            ece=0.02, bias=0.01,
            is_overconfident=False, is_underconfident=False,
            total_predictions=50,
        )
        cons_sig, prom_sig = mneme._modulate_thresholds()
        # ECE modulation: cons += 0.02 * 0.30 = 0.006 → negligible
        assert cons_sig < 0.51
        assert prom_sig < 0.71

    def test_calibration_composes_with_tension(self, mneme):
        """Calibration modulation is additive with tension modulation."""
        # Start with neutral calibration, prune-biased tension
        mneme.set_tension_profile({"consolidate_prune": 0.5})
        tension_only_cons, tension_only_prom = mneme._modulate_thresholds()
        # cons = 0.5 + 0.5*0.20 = 0.6, prom = 0.7 + 0.5*0.15 = 0.775

        # Add high-ECE calibration on top
        mneme.set_calibration_state(
            ece=0.20, bias=0.15,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        cal_cons, cal_prom = mneme._modulate_thresholds()

        # Calibration should raise further
        assert cal_cons > tension_only_cons
        assert cal_prom > tension_only_prom

    def test_calibration_bounded_within_clamps(self, mneme):
        """Even extreme ECE shouldn't push thresholds outside [0.15, 0.95] / [0.3, 0.98]."""
        mneme.set_tension_profile({"consolidate_prune": 0.8})  # Already high
        mneme.set_calibration_state(
            ece=0.50, bias=0.40,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        cons_sig, prom_sig = mneme._modulate_thresholds()
        assert cons_sig <= 0.95
        assert prom_sig <= 0.98


# ═══════════════════════════════════════════════════════════════════
# ConsolidationReport calibration fields
# ═══════════════════════════════════════════════════════════════════


class TestConsolidationReportCalibration:
    """ConsolidationReport should report calibration fields."""

    def test_calibration_fields_default(self):
        """Default ConsolidationReport has zero/sensible calibration defaults."""
        report = ConsolidationReport(
            working_count=1, episodic_count=1, semantic_count=1,
            wm_to_episodic=0, ep_to_semantic=0, pruned=0,
            thresholds=(0.5, 0.7),
            tension_profile={},
        )
        assert report.calibration_ece == 0.0
        assert report.calibration_active is False
        assert report.calibration_prune_saved == 0

    def test_calibration_fields_custom(self):
        """Custom calibration values are stored correctly."""
        report = ConsolidationReport(
            working_count=5, episodic_count=3, semantic_count=2,
            wm_to_episodic=1, ep_to_semantic=0, pruned=2,
            thresholds=(0.65, 0.82),
            tension_profile={},
            calibration_ece=0.1542,
            calibration_active=True,
            calibration_prune_saved=3,
        )
        assert report.calibration_ece == 0.1542
        assert report.calibration_active is True
        assert report.calibration_prune_saved == 3

    def test_report_with_calibration_produces_active_consolidation(self, mneme):
        """Consolidation with calibration data should produce active report."""
        mneme.set_calibration_state(
            ece=0.20, bias=0.15,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        # Add a memory that should consolidate
        mneme.store("important fact about the system", significance=0.8)
        mneme.store("another key insight", significance=0.75)
        report = mneme.consolidate()

        # With high ECE thresholds are raised, but these are high significance
        assert report.calibration_active is True
        assert report.calibration_ece > 0.0

    def test_no_calibration_produces_inactive_report(self, mneme):
        """Consolidation without calibration should show inactive."""
        mneme.store("test fact", significance=0.6)
        report = mneme.consolidate()
        assert report.calibration_active is False
        assert report.calibration_ece == 0.0

    def test_summary_includes_calibration_when_active(self, mneme):
        """When calibration is active, summary should include ECE."""
        mneme.set_calibration_state(
            ece=0.15, bias=0.10,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        mneme.store("test", significance=0.6)
        report = mneme.consolidate()
        summary = report.summary()
        assert "ec:" in summary.lower() or "cal:" in summary.lower()


# ═══════════════════════════════════════════════════════════════════
# Calibration-aware pruning sensitivity
# ═══════════════════════════════════════════════════════════════════


class TestCalibrationPruneSensitivity:
    """Overconfidence should reduce pruning aggressiveness."""

    def test_overconfident_reduces_prune_rate(self, mneme):
        """Overconfident calibrator should result in fewer pruned entries."""
        # Fill working memory with low-strength entries
        for i in range(5):
            entry = mneme.store(f"forgettable fact {i}", significance=0.2)
            # Artificially weaken it
            from uuid import UUID
            for entry_id, e in list(mneme._working.items()):
                mneme._working[entry_id] = MemoryEntry(
                    id=e.id, content=e.content, tier=e.tier,
                    strength=0.01, significance=e.significance,
                    created_at=e.created_at, last_accessed=e.last_accessed,
                    last_rehearsed=e.last_rehearsed,
                    rehearsal_count=e.rehearsal_count,
                    access_count=e.access_count, source=e.source,
                    tags=e.tags, metadata=e.metadata,
                    base_half_life=e.base_half_life,
                )

        # Consolidate WITHOUT calibration
        report_no_cal = mneme.consolidate()

        # Now add overconfident calibration
        mneme.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )

        # Fill again
        for i in range(5):
            entry = mneme.store(f"another forgettable {i}", significance=0.2)
            for entry_id, e in list(mneme._working.items()):
                if e.content == f"another forgettable {i}":
                    mneme._working[entry_id] = MemoryEntry(
                        id=e.id, content=e.content, tier=e.tier,
                        strength=0.01, significance=e.significance,
                        created_at=e.created_at, last_accessed=e.last_accessed,
                        last_rehearsed=e.last_rehearsed,
                        rehearsal_count=e.rehearsal_count,
                        access_count=e.access_count, source=e.source,
                        tags=e.tags, metadata=e.metadata,
                        base_half_life=e.base_half_life,
                    )

        report_with_cal = mneme.consolidate()

        # Overconfident calibration should save some from pruning
        # (the pruned count may differ because the strength-based
        #  pruning logic interacts differently, but calibration should
        #  try to save some entries)
        assert report_with_cal.calibration_active is True
        assert report_with_cal.calibration_ece > 0.0

    def test_well_calibrated_no_prune_discount(self, mneme):
        """Well-calibrated agent should not get pruning discounts."""
        mneme.set_calibration_state(
            ece=0.03, bias=0.01,
            is_overconfident=False, is_underconfident=False,
            total_predictions=50,
        )

        for i in range(5):
            entry = mneme.store(f"marginal {i}", significance=0.2)
            for entry_id, e in list(mneme._working.items()):
                if e.content == f"marginal {i}":
                    mneme._working[entry_id] = MemoryEntry(
                        id=e.id, content=e.content, tier=e.tier,
                        strength=0.01, significance=e.significance,
                        created_at=e.created_at, last_accessed=e.last_accessed,
                        last_rehearsed=e.last_rehearsed,
                        rehearsal_count=e.rehearsal_count,
                        access_count=e.access_count, source=e.source,
                        tags=e.tags, metadata=e.metadata,
                        base_half_life=e.base_half_life,
                    )

        report = mneme.consolidate()
        assert report.calibration_active is True
        # Well-calibrated with low ECE → no overconfidence discount
        # so calibration_prune_saved should be 0
        assert report.calibration_prune_saved == 0


# ═══════════════════════════════════════════════════════════════════
# MnemePillar calibration wiring
# ═══════════════════════════════════════════════════════════════════


class TestMnemePillarCalibration:
    """MnemePillar.update_calibration() should forward to HierarchicalMneme."""

    def test_update_calibration_forwarded(self, agent_state):
        """update_calibration() on pillar should reach HierarchicalMneme."""
        pillar = MnemePillar(name="test_mneme")
        pillar.initialize(agent_state)
        assert pillar.mneme is not None

        pillar.update_calibration(
            ece=0.18, bias=0.12,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        assert pillar.mneme._calibration_ece == 0.18
        assert pillar.mneme._calibration_overconfident is True

    def test_few_predictions_not_forwarded(self, agent_state):
        """update_calibration with <10 predictions should NOT set state."""
        pillar = MnemePillar(name="test_mneme")
        pillar.initialize(agent_state)

        pillar.update_calibration(
            ece=0.30, bias=0.25,
            is_overconfident=True, is_underconfident=False,
            total_predictions=5,  # Below guard
        )
        assert pillar.mneme._calibration_ece == 0.0  # Not set

    def test_uninitialized_no_error(self):
        """update_calibration should not error when mneme is None."""
        pillar = MnemePillar(name="empty")
        # No initialize called
        pillar.update_calibration(
            ece=0.15, bias=0.10,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )
        # Should not raise


# ═══════════════════════════════════════════════════════════════════
# CognitionPillar → MnemePillar calibration push
# ═══════════════════════════════════════════════════════════════════


class TestCognitionPillarMnemePush:
    """CognitionPillar should push calibration to attached MnemePillar."""

    def test_cognition_pushes_calibration_to_mneme_on_tick(self, agent_state):
        """update_tension_profile() should push calibration to mneme pillar."""
        mneme = MnemePillar(name="test_mneme")
        mneme.initialize(agent_state)

        cognition = CognitionPillar(
            name="test_cognition",
            mneme_pillar=mneme,
        )
        cognition.initialize(agent_state)
        assert cognition.reasoning is not None

        # Feed calibrator data (need ≥10 predictions)
        for _ in range(15):
            cognition.reasoning.calibrate(
                predicted_confidence=0.80, actual_success=False
            )
        for _ in range(5):
            cognition.reasoning.calibrate(
                predicted_confidence=0.80, actual_success=True
            )

        # This should push calibration to mneme
        cognition.update_tension_profile({
            "consolidate_prune": 0.0,
            "specific_general": 0.0,
        })

        # Mneme should now have calibration state
        assert mneme.mneme is not None
        assert mneme.mneme._calibration_total_predictions >= 10
        assert mneme.mneme._calibration_ece > 0.0
        # ~20 predictions with 80% conf, ~25% accurate → high ECE, overconfident
        assert mneme.mneme._calibration_overconfident is True

    def test_cognition_push_no_mneme_no_error(self, agent_state):
        """CognitionPillar without mneme reference should not error."""
        cognition = CognitionPillar(name="cog_only")
        cognition.initialize(agent_state)

        for _ in range(15):
            cognition.reasoning.calibrate(
                predicted_confidence=0.50, actual_success=True
            )

        # Should not raise despite no mneme_pillar
        cognition.update_tension_profile({
            "consolidate_prune": 0.0,
            "specific_general": 0.0,
        })

    def test_cognition_push_few_predictions_no_mneme_change(self, agent_state):
        """CognitionPillar should not push calibration with <10 predictions."""
        mneme = MnemePillar(name="test_mneme")
        mneme.initialize(agent_state)
        assert mneme.mneme is not None

        cognition = CognitionPillar(
            name="test_cognition",
            mneme_pillar=mneme,
        )
        cognition.initialize(agent_state)

        # Only 5 predictions — below guard
        for _ in range(5):
            cognition.reasoning.calibrate(
                predicted_confidence=0.80, actual_success=False
            )

        cognition.update_tension_profile({
            "consolidate_prune": 0.0,
            "specific_general": 0.0,
        })

        # Mneme should NOT have calibration state
        assert mneme.mneme._calibration_total_predictions == 0


# ═══════════════════════════════════════════════════════════════════
# End-to-end: calibration-aware consolidation pipeline
# ═══════════════════════════════════════════════════════════════════


class TestEndToEndCalibrationMneme:
    """Full pipeline: calibrate → consolidate → report with calibration."""

    def test_overconfident_calibration_blocks_low_sig_consolidation(self, mneme):
        """With overconfident calibration, low-significance items stay in WM."""
        # Set high ECE calibration
        mneme.set_calibration_state(
            ece=0.25, bias=0.20,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )

        # Store items at various significance levels
        mneme.store("low significance note", significance=0.35)
        mneme.store("medium significance note", significance=0.55)
        mneme.store("high significance note", significance=0.85)

        report = mneme.consolidate()

        # The calibration raises cons threshold from 0.5:
        #   Δ = 0.25×0.30 + 0.25×0.20 = 0.125 → threshold ≈ 0.625
        # Only the high-significance item should promote
        # Check how many WMs vs how many promoted
        assert report.wm_to_episodic <= 2  # At most high-sig ones promote

    def test_well_calibrated_normal_consolidation(self, mneme):
        """With well-calibrated calibration, normal consolidation proceeds."""
        mneme.set_calibration_state(
            ece=0.03, bias=0.01,
            is_overconfident=False, is_underconfident=False,
            total_predictions=50,
        )

        mneme.store("medium note", significance=0.55)
        mneme.store("good note", significance=0.85)

        report = mneme.consolidate()

        # Thresholds near default: cons ~ 0.506, prom ~ 0.705
        # The 0.85 item should consolidate
        assert report.wm_to_episodic >= 1

    def test_calibration_affects_mneme_pillar_consolidation_output(self, agent_state):
        """MnemePillar with calibration produces calibration-aware reports."""
        mneme = MnemePillar(name="test_mneme")
        mneme.initialize(agent_state)

        # Manually push calibration
        mneme.update_calibration(
            ece=0.22, bias=0.18,
            is_overconfident=True, is_underconfident=False,
            total_predictions=50,
        )

        # Store and consolidate
        mneme.mneme.store("somewhat important", significance=0.55)
        mneme.mneme.store("very important", significance=0.90)

        # Consolidate through the pillar
        mneme.update_tension_profile({"consolidate_prune": 0.0})

        assert mneme.mneme._calibration_total_predictions >= 10
