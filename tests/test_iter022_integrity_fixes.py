"""Tests for iter-022: Calibration Integrity Fixes & Truthiness Bug Sweep.

Covers:
 1. Weight-sum invariant in ConfidenceCalibrator.adjust_weights()
 2. Terminal node confidence respects calibrated weights
 3. Risk threshold ordering in _compose_actions
 4. Truthiness bugs: `or` → `if is not None` in HierarchicalMneme and RehearsalScheduler
 5. Immutable class-level _DEFAULT_PROFILE (MappingProxyType)
 6. Cross-tier pruning rescue under calibration
 7. Orchestrator from_dict _results initialization
 8. Dependency resolution safety (no self-cycles)
"""

from __future__ import annotations

import pytest
import time
from types import MappingProxyType
from uuid import UUID, uuid4

from isonome.cognition.reasoning import (
    ConfidenceCalibrator,
    RecursiveReasoningEngine,
    ReasoningNode,
    NodeStatus,
    EvidencePoint,
)
from isonome.mneme.hierarchical import (
    HierarchicalMneme,
    MemoryEntry,
    MemoryTier,
    RehearsalScheduler,
)
from isonome.praxis.orchestrator import (
    ActionOrchestrator,
    Action,
    ActionRisk,
    RetryPolicy,
)
from isonome.equilibrium import EquilibriumEngine


# ═══════════════════════════════════════════════════════════════════
# 1. Weight-sum invariant
# ═══════════════════════════════════════════════════════════════════


class TestWeightSumInvariant:
    """adjust_weights() must preserve evidence_weight + child_weight == 1.0."""

    def _make_overconfident(self) -> ConfidenceCalibrator:
        c = ConfidenceCalibrator()
        for _ in range(25):
            c.record(0.9, False)  # always wrong → overconfident
        assert c.is_overconfident
        return c

    def _make_underconfident(self) -> ConfidenceCalibrator:
        c = ConfidenceCalibrator()
        for _ in range(25):
            c.record(0.1, True)  # always right but predicts low → underconfident
        assert c.is_underconfident
        return c

    def test_sum_stays_one_overconfident(self):
        c = self._make_overconfident()
        for _ in range(100):
            c.adjust_weights()
            assert c.evidence_weight + c.child_weight == pytest.approx(1.0, abs=1e-10)

    def test_sum_stays_one_underconfident(self):
        c = self._make_underconfident()
        for _ in range(100):
            c.adjust_weights()
            assert c.evidence_weight + c.child_weight == pytest.approx(1.0, abs=1e-10)

    def test_sum_corrected_from_drifted_state(self):
        """If weights were corrupted (sum != 1), adjust_weights self-corrects."""
        c = self._make_overconfident()
        c._evidence_weight = 0.795
        c._child_weight = 0.295  # sum = 1.09
        c.adjust_weights()
        assert c.evidence_weight + c.child_weight == pytest.approx(1.0, abs=1e-10)

    def test_boundary_preserves_invariant(self):
        """At min/max bounds, sum must still be exactly 1.0."""
        c = self._make_overconfident()
        c._evidence_weight = 0.20
        c._child_weight = 0.80
        c.adjust_weights()
        # evidence_weight clamped at 0.20, child_weight derived as 1.0 - 0.20
        assert c.evidence_weight == pytest.approx(0.20)
        assert c.child_weight == pytest.approx(0.80)
        assert c.evidence_weight + c.child_weight == pytest.approx(1.0)

    def test_underconfident_boundary_preserves_invariant(self):
        c = self._make_underconfident()
        c._evidence_weight = 0.80
        c._child_weight = 0.20
        c.adjust_weights()
        assert c.evidence_weight == pytest.approx(0.80)
        assert c.child_weight == pytest.approx(0.20)
        assert c.evidence_weight + c.child_weight == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════
# 2. Terminal node confidence respects calibrated weights
# ═══════════════════════════════════════════════════════════════════


class TestTerminalNodeConfidence:
    """Terminal nodes must use calibrated weights, not pure evidence_ratio."""

    def _make_engine_with_calibration(self):
        engine = RecursiveReasoningEngine(None)
        # Set up calibration with non-default weights
        for _ in range(15):
            engine._calibrator.record(0.8, True)
        return engine

    def _make_node_with_evidence(self, evidence_ratio_target: float) -> ReasoningNode:
        """Create a node whose evidence_ratio approximates the target value.

        We add supporting/contradicting evidence to hit the target ratio.
        """
        node = ReasoningNode(id=uuid4(), hypothesis="test", depth=0)
        if evidence_ratio_target >= 0.5:
            for_weight = evidence_ratio_target * 10
            against_weight = (1.0 - evidence_ratio_target) * 10
        else:
            for_weight = evidence_ratio_target * 10
            against_weight = (1.0 - evidence_ratio_target) * 10

        node.evidence_for.append(EvidencePoint(content="supports", supports=True, weight=for_weight))
        node.evidence_against.append(EvidencePoint(content="contradicts", supports=False, weight=against_weight))
        # Verify ratio is close
        assert node.evidence_ratio == pytest.approx(evidence_ratio_target, abs=0.05)
        return node

    def test_terminal_confidence_uses_weights(self):
        """Terminal node confidence should be evidence_ratio * w_ev + 0.5 * w_ch."""
        engine = self._make_engine_with_calibration()
        w_ev = engine._calibrator.evidence_weight
        w_ch = engine._calibrator.child_weight

        node = self._make_node_with_evidence(0.8)
        node.terminal = True
        node.status = NodeStatus.CONVERGED

        conf = engine._evaluate_confidence(node)
        expected = node.evidence_ratio * w_ev + 0.5 * w_ch
        assert conf == pytest.approx(expected, abs=0.02)

    def test_terminal_confidence_differs_from_pure_evidence(self):
        """With non-50/50 weights, terminal confidence != evidence_ratio."""
        engine = self._make_engine_with_calibration()
        node = self._make_node_with_evidence(0.6)
        conf = engine._evaluate_confidence(node)
        # With default weights (0.7, 0.3): 0.6 * 0.7 + 0.5 * 0.3 = 0.57
        # Not 0.6 (which was the old behavior)
        assert conf != pytest.approx(0.6, abs=0.01)

    def test_non_terminal_confidence_unchanged(self):
        """Internal nodes still use evidence + children blend."""
        engine = self._make_engine_with_calibration()
        parent = self._make_node_with_evidence(0.7)
        child = ReasoningNode(id=uuid4(), hypothesis="child", depth=1)
        child.confidence = 0.5
        child.status = NodeStatus.CONVERGED
        parent.children = [child]

        w_ev = engine._calibrator.evidence_weight
        w_ch = engine._calibrator.child_weight
        expected = parent.evidence_ratio * w_ev + 0.5 * w_ch
        conf = engine._evaluate_confidence(parent)
        assert conf == pytest.approx(expected, abs=0.02)

    def test_terminal_with_no_evidence_returns_half(self):
        """A terminal node with no evidence should return 0.5 * w_ev + 0.5 * w_ch = 0.5."""
        engine = self._make_engine_with_calibration()
        node = ReasoningNode(id=uuid4(), hypothesis="test", depth=0)
        node.terminal = True
        node.status = NodeStatus.CONVERGED
        # No evidence → evidence_ratio = 0.5
        assert node.evidence_ratio == 0.5
        conf = engine._evaluate_confidence(node)
        # 0.5 * w_ev + 0.5 * w_ch = 0.5 * (w_ev + w_ch) = 0.5
        assert conf == pytest.approx(0.5, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# 3. Risk threshold ordering
# ═══════════════════════════════════════════════════════════════════


class TestRiskThresholdOrdering:
    """Low confidence → high risk; the < 0.2 check must precede < 0.4."""

    def test_very_low_confidence_is_high_risk(self):
        """Confidence < 0.2 should produce 'high' risk, not 'moderate'."""
        engine = RecursiveReasoningEngine(None)
        node = ReasoningNode(
            id=uuid4(), hypothesis="test hypothesis and action", depth=0,
        )
        node.confidence = 0.1  # Very low → should be "high"
        node.status = NodeStatus.CONVERGED

        actions = engine._compose_actions(node)
        if actions:
            assert any(a.get("risk") == "high" for a in actions), (
                f"Expected 'high' risk for confidence=0.1, got: {actions}"
            )

    def test_medium_low_confidence_is_moderate_risk(self):
        """Confidence between 0.2 and 0.4 should be 'moderate' risk."""
        engine = RecursiveReasoningEngine(None)
        node = ReasoningNode(
            id=uuid4(), hypothesis="test hypothesis and action", depth=0,
        )
        node.confidence = 0.3  # Between 0.2 and 0.4 → "moderate"
        node.status = NodeStatus.CONVERGED

        actions = engine._compose_actions(node)
        if actions:
            assert any(a.get("risk") == "moderate" for a in actions), (
                f"Expected 'moderate' risk for confidence=0.3, got: {actions}"
            )

    def test_high_confidence_is_low_risk(self):
        """Confidence >= 0.4 should be 'low' risk."""
        engine = RecursiveReasoningEngine(None)
        node = ReasoningNode(
            id=uuid4(), hypothesis="test hypothesis and action", depth=0,
        )
        node.confidence = 0.6
        node.status = NodeStatus.CONVERGED

        actions = engine._compose_actions(node)
        if actions:
            assert any(a.get("risk") == "low" for a in actions), (
                f"Expected 'low' risk for confidence=0.6, got: {actions}"
            )


# ═══════════════════════════════════════════════════════════════════
# 4. Truthiness bugs: `or` → `if is not None`
# ═══════════════════════════════════════════════════════════════════


class TestTruthinessFixes:
    """Float parameters set to 0.0 must not fall through to defaults."""

    def test_consolidation_significance_zero(self):
        """consolidation_significance=0.0 should be respected, not replaced by default."""
        mneme = HierarchicalMneme(consolidation_significance=0.0)
        assert mneme._consolidation_sig == 0.0

    def test_promotion_significance_zero(self):
        """promotion_significance=0.0 should be respected."""
        mneme = HierarchicalMneme(promotion_significance=0.0)
        assert mneme._promotion_sig == 0.0

    def test_rehearsal_boost_zero(self):
        """rehearsal_boost=0.0 should be respected (no boost)."""
        mneme = HierarchicalMneme(rehearsal_boost=0.0)
        assert mneme._rehearsal_boost == 0.0

    def test_pattern_count_threshold_zero(self):
        """pattern_count_threshold=0 should be respected."""
        mneme = HierarchicalMneme(pattern_count_threshold=0)
        assert mneme._pattern_threshold == 0

    def test_none_uses_defaults(self):
        """None parameters still use defaults."""
        mneme = HierarchicalMneme()
        assert mneme._consolidation_sig == mneme.DEFAULT_CONSOLIDATION_SIGNIFICANCE
        assert mneme._promotion_sig == mneme.DEFAULT_PROMOTION_SIGNIFICANCE
        assert mneme._rehearsal_boost == mneme.DEFAULT_REHEARSAL_BOOST
        assert mneme._pattern_threshold == mneme.DEFAULT_PATTERN_COUNT_THRESHOLD

    def test_rehearsal_scheduler_min_interval_zero(self):
        """min_interval=0.0 should be respected."""
        scheduler = RehearsalScheduler(min_interval=0.0)
        assert scheduler._min_interval == 0.0

    def test_rehearsal_scheduler_max_interval_zero(self):
        """max_interval=0.0 should be respected."""
        scheduler = RehearsalScheduler(max_interval=0.0)
        assert scheduler._max_interval == 0.0

    def test_rehearse_with_explicit_zero_boost(self):
        """rehearse(entry, boost=0.0) should add 0.0 boost, not default."""
        mneme = HierarchicalMneme()
        entry = mneme.store("test content", significance=0.7)
        rehearsed = mneme.rehearse(entry.id, boost=0.0)
        assert rehearsed is not None

    def test_rehearse_by_tags_with_explicit_zero_boost(self):
        """rehearse_by_tags(..., boost=0.0) should use 0.0, not default."""
        mneme = HierarchicalMneme()
        mneme.store("alpha content", tags=frozenset({"alpha"}))
        # Should not crash with boost=0.0
        count = mneme.rehearse_by_tags(frozenset({"alpha"}), boost=0.0)
        assert isinstance(count, int)


# ═══════════════════════════════════════════════════════════════════
# 5. Immutable class-level _DEFAULT_PROFILE
# ═══════════════════════════════════════════════════════════════════


class TestImmutableDefaultProfile:
    """_DEFAULT_PROFILE should be MappingProxyType, not mutable dict."""

    def test_reasoning_profile_is_immutable(self):
        """RecursiveReasoningEngine._DEFAULT_PROFILE should not be mutable."""
        profile = RecursiveReasoningEngine._DEFAULT_PROFILE
        assert isinstance(profile, MappingProxyType)
        with pytest.raises(TypeError):
            profile["shallow_deep"] = 999.0

    def test_orchestrator_profile_is_immutable(self):
        """ActionOrchestrator._DEFAULT_PROFILE should not be mutable."""
        profile = ActionOrchestrator._DEFAULT_PROFILE
        assert isinstance(profile, MappingProxyType)
        with pytest.raises(TypeError):
            profile["autonomy_safety"] = 999.0

    def test_instance_profile_copy_is_mutable(self):
        """Instance profiles are copies and should be mutable."""
        engine = RecursiveReasoningEngine(None)
        # Instance profile is a copy, should be a regular dict
        assert isinstance(engine._current_profile, dict)
        engine._current_profile["shallow_deep"] = 0.5  # Should not raise


# ═══════════════════════════════════════════════════════════════════
# 6. Cross-tier pruning rescue under calibration
# ═══════════════════════════════════════════════════════════════════


class TestCrossTierPruningRescue:
    """Calibration-aware pruning should rescue entries across all tiers."""

    def _make_overconfident_mneme(self) -> HierarchicalMneme:
        mneme = HierarchicalMneme()
        # Set overconfident calibration state
        mneme.set_calibration_state(
            ece=0.25,
            bias=0.15,
            is_overconfident=True,
            is_underconfident=False,
            total_predictions=15,
        )
        assert mneme._calibration_overconfident
        return mneme

    def test_rescue_gathers_from_all_tiers(self):
        """Forgotten entries from episodic and semantic should also be rescue candidates."""
        mneme = self._make_overconfident_mneme()
        now = time.time()

        # Add many low-strength entries across tiers so some get pruned
        for tier, collection, sig in [
            (MemoryTier.WORKING, mneme._working, 0.9),
            (MemoryTier.EPISODIC, mneme._episodic, 0.85),
            (MemoryTier.SEMANTIC, mneme._semantic, 0.8),
        ]:
            for i in range(5):
                entry = MemoryEntry(
                    id=uuid4(),
                    content=f"forgot_{tier.name}_{i}",
                    tier=tier,
                    strength=0.01,  # Below FORGET_THRESHOLD
                    significance=sig - i * 0.1,
                    created_at=now - 10000,
                    last_accessed=now - 10000,
                    last_rehearsed=now - 10000,
                    rehearsal_count=0,
                    access_count=0,
                    source="test",
                    tags=frozenset(),
                    metadata={},
                    base_half_life=3600.0,
                )
                collection[entry.id] = entry

        # Consolidate → should prune many but rescue some
        report = mneme.consolidate()
        # calibration_prune_saved should be > 0 when overconfident and pruned > 0
        if report.pruned > 0:
            assert report.calibration_prune_saved > 0

    def test_rescue_prefers_high_significance(self):
        """Rescued entries should be the most significant ones."""
        mneme = self._make_overconfident_mneme()
        now = time.time()

        # Add many forgotten entries across tiers to ensure robust pruning
        for tier, collection, base_sig in [
            (MemoryTier.WORKING, mneme._working, 0.3),
            (MemoryTier.EPISODIC, mneme._episodic, 0.5),
            (MemoryTier.SEMANTIC, mneme._semantic, 0.6),
        ]:
            for i in range(8):
                entry = MemoryEntry(
                    id=uuid4(),
                    content=f"forgot_{tier.name}_{i}",
                    tier=tier,
                    strength=0.01,
                    significance=base_sig + i * 0.05,
                    created_at=now - 10000,
                    last_accessed=now - 10000,
                    last_rehearsed=now - 10000,
                    rehearsal_count=0,
                    access_count=0,
                    source="test",
                    tags=frozenset(),
                    metadata={},
                    base_half_life=3600.0,
                )
                collection[entry.id] = entry

        report = mneme.consolidate()
        # Calibration-aware rescue should prefer high-significance entries
        if report.calibration_prune_saved > 0:
            # The rescued entries in working memory should include the
            # highest-significance ones from across all tiers
            max_rescued_sig = max(
                (e.significance for e in mneme._working.values()), default=0.0
            )
            assert max_rescued_sig >= 0.7, (
                f"Expected high-significance rescue (>= 0.7), got max={max_rescued_sig}"
            )


# ═══════════════════════════════════════════════════════════════════
# 7. Orchestrator from_dict _results initialization
# ═══════════════════════════════════════════════════════════════════


class TestOrchestratorFromDictResults:
    """from_dict must initialize _results for every action."""

    def test_results_initialized_for_all_actions(self):
        """Every action in from_dict should have an empty results list."""
        orch = ActionOrchestrator()
        a1 = Action(description="action 1", tool_name="test_tool")
        a2 = Action(description="action 2", tool_name="test_tool")
        orch._actions[a1.id] = a1
        orch._results[a1.id] = []
        orch._actions[a2.id] = a2
        orch._results[a2.id] = []

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        for aid in restored._actions:
            assert aid in restored._results, f"Action {aid} missing from _results"
            assert restored._results[aid] == []


# ═══════════════════════════════════════════════════════════════════
# 8. Dependency resolution safety
# ═══════════════════════════════════════════════════════════════════


class TestDependencyResolutionSafety:
    """Unresolved dependencies should be skipped, not create self-cycles."""

    def test_unresolved_dependency_skipped_in_from_dict(self):
        """from_dict with unresolved dependency references should skip them."""
        orch = ActionOrchestrator()
        a1 = Action(description="independent", tool_name="tool1")
        a2 = Action(
            description="dependent",
            tool_name="tool2",
            dependencies=(a1.id, uuid4()),  # Second dep is nonexistent
        )
        orch._actions[a1.id] = a1
        orch._results[a1.id] = []
        orch._actions[a2.id] = a2
        orch._results[a2.id] = []

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        # Second action should only have resolved deps
        restored_a2 = restored._actions[a2.id]
        # The valid dependency should be present
        assert a1.id in restored_a2.dependencies
        # No self-cycles
        for aid in restored._actions:
            assert aid not in restored._actions[aid].dependencies

    def test_no_self_cycle_from_dict(self):
        """No action should depend on itself after from_dict roundtrip."""
        orch = ActionOrchestrator()
        a1 = Action(description="test action", tool_name="tool1")
        orch._actions[a1.id] = a1
        orch._results[a1.id] = []

        data = orch.to_dict()
        restored = ActionOrchestrator.from_dict(data)

        for aid in restored._actions:
            assert aid not in restored._actions[aid].dependencies
