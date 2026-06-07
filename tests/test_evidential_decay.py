"""Tests for Evidential Decay — iteration-029.

Covers:
- evidential_decay() function: exponential decay, edge cases, validation
- EvidencePoint.created_at field: default, explicit, frozen
- ReasoningNode.evidence_ratio_at(): time-decayed evidence ratio
- ReasoningNode.decay_weights(): in-place weight replacement
- RecursiveReasoningEngine integration: half-life parameter,
  decay-aware confidence evaluation, backward compatibility
"""

import time
import uuid

import pytest

from isonome.cognition.reasoning import (
    ConfidenceCalibrator,
    EvidencePoint,
    NodeStatus,
    ReasoningNode,
    RecursiveReasoningEngine,
    evidential_decay,
)
from isonome.types import Pillar


# ── Helpers ──────────────────────────────────────────────────────


def _ep(
    content: str = "test",
    supports: bool = True,
    weight: float = 1.0,
    source: str = "test",
    created_at: float | None = None,
) -> EvidencePoint:
    """Create an EvidencePoint with optional created_at."""
    kwargs: dict = {
        "content": content,
        "supports": supports,
        "weight": weight,
        "source": source,
    }
    if created_at is not None:
        kwargs["created_at"] = created_at
    return EvidencePoint(**kwargs)


# ═══════════════════════════════════════════════════════════════════
# evidential_decay() Function
# ═══════════════════════════════════════════════════════════════════


class TestEvidentialDecayBasic:
    """Core decay computation tests."""

    def test_no_decay_at_creation(self):
        """Evidence at age 0 should retain full weight."""
        now = 1000.0
        decayed = evidential_decay(1.0, now, now=now, half_life=3600.0)
        assert decayed == 1.0

    def test_half_life_decay(self):
        """After one half-life, weight should be halved."""
        now = 4600.0  # 3600 seconds after creation
        decayed = evidential_decay(1.0, 1000.0, now=now, half_life=3600.0)
        assert abs(decayed - 0.5) < 1e-9

    def test_two_half_lives(self):
        """After two half-lives, weight should be quartered."""
        now = 8200.0  # 7200 seconds after creation (2 × 3600)
        decayed = evidential_decay(1.0, 1000.0, now=now, half_life=3600.0)
        assert abs(decayed - 0.25) < 1e-9

    def test_partial_weight_decay(self):
        """Decay applies proportionally to the original weight."""
        now = 4600.0
        decayed = evidential_decay(0.8, 1000.0, now=now, half_life=3600.0)
        assert abs(decayed - 0.4) < 1e-9

    def test_future_timestamp_no_decay(self):
        """Evidence created in the future (clock skew) should not decay."""
        now = 500.0
        created_at = 1000.0
        decayed = evidential_decay(1.0, created_at, now=now, half_life=3600.0)
        assert decayed == 1.0

    def test_zero_age_returns_full_weight(self):
        """Age == 0 should return the original weight."""
        decayed = evidential_decay(0.7, 1000.0, now=1000.0, half_life=60.0)
        assert decayed == 0.7


class TestEvidentialDecayMinWeight:
    """min_weight floor behavior."""

    def test_min_weight_respected(self):
        """Decayed weight should not fall below min_weight."""
        # 10 half-lives → weight ≈ 0.001
        now = 1000.0 + 10 * 3600.0
        decayed = evidential_decay(
            1.0, 1000.0, now=now, half_life=3600.0, min_weight=0.1
        )
        assert decayed == 0.1

    def test_min_weight_zero_default(self):
        """Default min_weight=0.0 allows full decay to zero."""
        now = 1000.0 + 50 * 3600.0  # 50 half-lives
        decayed = evidential_decay(1.0, 1000.0, now=now, half_life=3600.0)
        assert decayed < 1e-10

    def test_min_weight_equal_to_weight(self):
        """min_weight == weight should return weight immediately."""
        decayed = evidential_decay(
            0.5, 1000.0, now=5000.0, half_life=3600.0, min_weight=0.5
        )
        assert decayed == 0.5

    def test_min_weight_greater_than_weight_raises(self):
        """min_weight > weight is invalid and should raise."""
        with pytest.raises(ValueError, match="cannot exceed"):
            evidential_decay(
                0.3, 1000.0, now=2000.0, half_life=3600.0, min_weight=0.5
            )

    def test_negative_min_weight_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            evidential_decay(
                1.0, 1000.0, now=2000.0, half_life=3600.0, min_weight=-0.1
            )


class TestEvidentialDecayValidation:
    """Input validation for evidential_decay."""

    def test_zero_half_life_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            evidential_decay(1.0, 1000.0, now=2000.0, half_life=0)

    def test_negative_half_life_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            evidential_decay(1.0, 1000.0, now=2000.0, half_life=-100.0)

    def test_infinite_half_life_no_decay(self):
        """inf half-life should disable decay entirely."""
        decayed = evidential_decay(
            1.0, 1000.0, now=9999999.0, half_life=float("inf")
        )
        assert decayed == 1.0

    def test_very_small_half_life(self):
        """Very small half-life should decay rapidly."""
        now = 1000.0 + 1.0  # 1 second after
        decayed = evidential_decay(1.0, 1000.0, now=now, half_life=0.001)
        assert decayed < 1e-10

    def test_now_defaults_to_current_time(self):
        """When now is None, it should use time.time()."""
        created_at = time.time() - 60  # 1 minute ago
        decayed = evidential_decay(1.0, created_at, half_life=3600.0)
        # Should be slightly less than 1.0
        assert 0.95 < decayed <= 1.0


class TestEvidentialDecayMonotonicity:
    """Decay should be monotonically decreasing with age."""

    def test_older_evidence_decays_more(self):
        created_at = 1000.0
        w1 = evidential_decay(1.0, created_at, now=2000.0, half_life=3600.0)
        w2 = evidential_decay(1.0, created_at, now=3000.0, half_life=3600.0)
        w3 = evidential_decay(1.0, created_at, now=5000.0, half_life=3600.0)
        assert w1 > w2 > w3


# ═══════════════════════════════════════════════════════════════════
# EvidencePoint.created_at
# ═══════════════════════════════════════════════════════════════════


class TestEvidencePointCreatedAt:
    """created_at field on EvidencePoint."""

    def test_default_is_current_time(self):
        """Default created_at should be approximately time.time()."""
        before = time.time()
        ep = EvidencePoint(content="test", supports=True, weight=1.0, source="test")
        after = time.time()
        assert before <= ep.created_at <= after

    def test_explicit_created_at(self):
        """Explicit created_at should be stored exactly."""
        ts = 1700000000.0
        ep = _ep(created_at=ts)
        assert ep.created_at == ts

    def test_frozen_immutability(self):
        """EvidencePoint is frozen — created_at cannot be changed."""
        ep = _ep(created_at=1000.0)
        with pytest.raises(AttributeError):
            ep.created_at = 2000.0  # type: ignore[misc]

    def test_created_at_preserved_in_decay_weights(self):
        """decay_weights() should preserve created_at in new objects."""
        ep = _ep(weight=1.0, created_at=1000.0)
        node = ReasoningNode(hypothesis="test", depth=0, evidence_for=[ep])
        node.decay_weights(now=4600.0, half_life=3600.0)
        assert node.evidence_for[0].created_at == 1000.0

    def test_id_preserved_in_decay_weights(self):
        """decay_weights() should preserve id in new objects."""
        ep = _ep(weight=1.0, created_at=1000.0)
        original_id = ep.id
        node = ReasoningNode(hypothesis="test", depth=0, evidence_for=[ep])
        node.decay_weights(now=4600.0, half_life=3600.0)
        assert node.evidence_for[0].id == original_id


# ═══════════════════════════════════════════════════════════════════
# ReasoningNode.evidence_ratio_at()
# ═══════════════════════════════════════════════════════════════════


class TestEvidenceRatioAt:
    """Time-decayed evidence ratio on ReasoningNode."""

    def test_fresh_evidence_matches_standard_ratio(self):
        """With zero age, evidence_ratio_at should match evidence_ratio."""
        now = 1000.0
        ep_for = _ep(supports=True, weight=0.8, created_at=now)
        ep_against = _ep(supports=False, weight=0.2, created_at=now)
        node = ReasoningNode(
            hypothesis="test", depth=0,
            evidence_for=[ep_for], evidence_against=[ep_against],
        )
        assert node.evidence_ratio == pytest.approx(0.8)
        assert node.evidence_ratio_at(now=now, half_life=3600.0) == pytest.approx(0.8)

    def test_stale_for_evidence_reduces_ratio(self):
        """When 'for' evidence is stale, ratio should drop toward 0.5."""
        old_for = _ep(supports=True, weight=1.0, created_at=1000.0)
        fresh_against = _ep(supports=False, weight=1.0, created_at=4000.0)
        node = ReasoningNode(
            hypothesis="test", depth=0,
            evidence_for=[old_for], evidence_against=[fresh_against],
        )
        now = 4600.0  # old_for is 3600s old (1 half-life), fresh_against is 600s old
        decayed = node.evidence_ratio_at(now=now, half_life=3600.0)
        standard = node.evidence_ratio
        # With decay, the 'for' weight drops to 0.5 while 'against' stays ≈1.0
        # So decayed ratio should be less than standard ratio
        assert decayed < standard

    def test_empty_evidence_returns_half(self):
        """No evidence should return 0.5 regardless of decay params."""
        node = ReasoningNode(hypothesis="test", depth=0)
        assert node.evidence_ratio_at(now=5000.0, half_life=3600.0) == 0.5

    def test_all_decayed_to_zero_returns_half(self):
        """When all weights decay to zero, ratio returns 0.5."""
        ep_for = _ep(supports=True, weight=1.0, created_at=1000.0)
        ep_against = _ep(supports=False, weight=1.0, created_at=1000.0)
        node = ReasoningNode(
            hypothesis="test", depth=0,
            evidence_for=[ep_for], evidence_against=[ep_against],
        )
        # 100 half-lives → both weights ≈ 0
        now = 1000.0 + 100 * 3600.0
        assert node.evidence_ratio_at(now=now, half_life=3600.0) == 0.5

    def test_min_weight_prevents_total_vanishing(self):
        """min_weight ensures some weight remains even for old evidence."""
        ep_for = _ep(supports=True, weight=1.0, created_at=1000.0)
        node = ReasoningNode(
            hypothesis="test", depth=0, evidence_for=[ep_for],
        )
        now = 1000.0 + 50 * 3600.0  # Very old
        decayed = node.evidence_ratio_at(now=now, half_life=3600.0, min_weight=0.1)
        # With min_weight=0.1, the for weight should be exactly 0.1
        assert decayed == pytest.approx(1.0)  # 0.1 / 0.1 = 1.0


# ═══════════════════════════════════════════════════════════════════
# ReasoningNode.decay_weights()
# ═══════════════════════════════════════════════════════════════════


class TestDecayWeights:
    """In-place weight replacement on ReasoningNode."""

    def test_fresh_evidence_unchanged(self):
        """Fresh evidence should have unchanged weight after decay."""
        now = 1000.0
        ep = _ep(supports=True, weight=0.9, created_at=now)
        node = ReasoningNode(hypothesis="test", depth=0, evidence_for=[ep])
        node.decay_weights(now=now, half_life=3600.0)
        assert node.evidence_for[0].weight == pytest.approx(0.9)

    def test_old_evidence_weight_reduced(self):
        """Old evidence should have reduced weight after decay."""
        ep = _ep(supports=True, weight=1.0, created_at=1000.0)
        node = ReasoningNode(hypothesis="test", depth=0, evidence_for=[ep])
        node.decay_weights(now=4600.0, half_life=3600.0)
        assert node.evidence_for[0].weight == pytest.approx(0.5)

    def test_against_evidence_also_decayed(self):
        """Decay should apply to evidence_against as well."""
        ep = _ep(supports=False, weight=1.0, created_at=1000.0)
        node = ReasoningNode(hypothesis="test", depth=0, evidence_against=[ep])
        node.decay_weights(now=4600.0, half_life=3600.0)
        assert node.evidence_against[0].weight == pytest.approx(0.5)

    def test_preserves_content(self):
        """decay_weights should preserve evidence content."""
        ep = _ep(content="important fact", supports=True, weight=1.0, created_at=1000.0)
        node = ReasoningNode(hypothesis="test", depth=0, evidence_for=[ep])
        node.decay_weights(now=4600.0, half_life=3600.0)
        assert node.evidence_for[0].content == "important fact"

    def test_preserves_source(self):
        """decay_weights should preserve evidence source."""
        ep = _ep(supports=True, weight=1.0, source="attention", created_at=1000.0)
        node = ReasoningNode(hypothesis="test", depth=0, evidence_for=[ep])
        node.decay_weights(now=4600.0, half_life=3600.0)
        assert node.evidence_for[0].source == "attention"

    def test_preserves_supports_flag(self):
        """decay_weights should preserve the supports/against flag."""
        ep_for = _ep(supports=True, weight=1.0, created_at=1000.0)
        ep_against = _ep(supports=False, weight=1.0, created_at=1000.0)
        node = ReasoningNode(
            hypothesis="test", depth=0,
            evidence_for=[ep_for], evidence_against=[ep_against],
        )
        node.decay_weights(now=4600.0, half_life=3600.0)
        assert node.evidence_for[0].supports is True
        assert node.evidence_against[0].supports is False

    def test_min_weight_applied(self):
        """min_weight should floor the decayed weight."""
        ep = _ep(supports=True, weight=1.0, created_at=1000.0)
        node = ReasoningNode(hypothesis="test", depth=0, evidence_for=[ep])
        node.decay_weights(
            now=1000.0 + 50 * 3600.0, half_life=3600.0, min_weight=0.1
        )
        assert node.evidence_for[0].weight == pytest.approx(0.1)

    def test_multiple_evidence_points(self):
        """decay_weights should handle multiple evidence points."""
        ep1 = _ep(supports=True, weight=1.0, created_at=1000.0)
        ep2 = _ep(supports=True, weight=0.6, created_at=2000.0)
        node = ReasoningNode(hypothesis="test", depth=0, evidence_for=[ep1, ep2])
        now = 4600.0
        node.decay_weights(now=now, half_life=3600.0)
        # ep1: age=3600s, weight=1.0 → 0.5
        # ep2: age=2600s, weight=0.6 → 0.6 * 2^(-2600/3600)
        expected1 = 1.0 * 0.5 ** (3600.0 / 3600.0)
        expected2 = 0.6 * 0.5 ** (2600.0 / 3600.0)
        assert node.evidence_for[0].weight == pytest.approx(expected1)
        assert node.evidence_for[1].weight == pytest.approx(expected2)

    def test_empty_evidence_lists(self):
        """decay_weights on a node with no evidence should not crash."""
        node = ReasoningNode(hypothesis="test", depth=0)
        node.decay_weights(now=5000.0, half_life=3600.0)
        assert node.evidence_for == []
        assert node.evidence_against == []

    def test_evidence_ratio_matches_after_decay(self):
        """After decay_weights, evidence_ratio should equal evidence_ratio_at."""
        ep_for = _ep(supports=True, weight=1.0, created_at=1000.0)
        ep_against = _ep(supports=False, weight=0.8, created_at=1100.0)
        node = ReasoningNode(
            hypothesis="test", depth=0,
            evidence_for=[ep_for], evidence_against=[ep_against],
        )
        now = 4600.0
        half_life = 3600.0
        # Compute ratio_at before decay
        ratio_at = node.evidence_ratio_at(now=now, half_life=half_life)
        # Apply decay
        node.decay_weights(now=now, half_life=half_life)
        # Standard ratio should now match the pre-decay ratio_at
        assert node.evidence_ratio == pytest.approx(ratio_at)


# ═══════════════════════════════════════════════════════════════════
# RecursiveReasoningEngine Integration
# ═══════════════════════════════════════════════════════════════════


class TestEngineEvidentialDecay:
    """Engine constructor and confidence evaluation with decay."""

    def test_default_half_life_is_infinite(self):
        """Default should be inf (no decay) — backward compatible."""
        engine = RecursiveReasoningEngine()
        assert engine.evidential_half_life == float("inf")

    def test_default_min_weight_is_zero(self):
        engine = RecursiveReasoningEngine()
        assert engine.evidential_min_weight == 0.0

    def test_custom_half_life(self):
        engine = RecursiveReasoningEngine(evidential_half_life=1800.0)
        assert engine.evidential_half_life == 1800.0

    def test_custom_min_weight(self):
        engine = RecursiveReasoningEngine(
            evidential_half_life=3600.0, evidential_min_weight=0.05
        )
        assert engine.evidential_min_weight == 0.05

    def test_zero_half_life_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            RecursiveReasoningEngine(evidential_half_life=0)

    def test_negative_half_life_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            RecursiveReasoningEngine(evidential_half_life=-100.0)

    def test_negative_min_weight_raises(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            RecursiveReasoningEngine(
                evidential_half_life=3600.0, evidential_min_weight=-0.1
            )

    def test_infinite_half_life_uses_standard_ratio(self):
        """With inf half-life, _evaluate_confidence should use standard ratio."""
        engine = RecursiveReasoningEngine(evidential_half_life=float("inf"))
        ep_for = _ep(supports=True, weight=0.8, created_at=1000.0)
        node = ReasoningNode(
            hypothesis="test", depth=0, evidence_for=[ep_for],
        )
        # With inf half-life, should use standard evidence_ratio
        conf = engine._evaluate_confidence(node)
        w_ev = engine.calibrator.evidence_weight
        w_ch = engine.calibrator.child_weight
        expected = node.evidence_ratio * w_ev + 0.5 * w_ch
        assert conf == pytest.approx(expected)

    def test_finite_half_life_uses_decayed_ratio(self):
        """With finite half-life, _evaluate_confidence should use decayed ratio."""
        engine = RecursiveReasoningEngine(evidential_half_life=3600.0)
        # Create node with old evidence
        old_ep = _ep(supports=True, weight=1.0, created_at=1000.0)
        node = ReasoningNode(hypothesis="test", depth=0, evidence_for=[old_ep])

        # We can't easily control 'now' inside _evaluate_confidence,
        # but we can verify it doesn't crash and produces a valid result
        conf = engine._evaluate_confidence(node)
        assert 0.0 <= conf <= 1.0

    def test_decay_confidence_lower_than_standard(self):
        """With stale evidence, decay-based confidence should be different."""
        # Engine with decay
        engine_decay = RecursiveReasoningEngine(evidential_half_life=0.01)
        # Engine without decay
        engine_no_decay = RecursiveReasoningEngine()

        old_ep = _ep(supports=True, weight=1.0, created_at=time.time() - 100)
        node = ReasoningNode(hypothesis="test", depth=0, evidence_for=[old_ep])

        conf_decay = engine_decay._evaluate_confidence(node)
        conf_no_decay = engine_no_decay._evaluate_confidence(node)

        # With very short half-life (0.01s) and 100s old evidence,
        # the decayed confidence should be much lower
        assert conf_decay < conf_no_decay


class TestEngineBackwardCompatibility:
    """Ensure existing behavior is preserved when decay is disabled."""

    def test_default_engine_no_decay(self):
        """Default engine should produce identical results to pre-decode."""
        engine = RecursiveReasoningEngine()
        ep_for = _ep(supports=True, weight=0.9, created_at=time.time() - 3600)
        ep_against = _ep(supports=False, weight=0.3, created_at=time.time() - 3600)
        node = ReasoningNode(
            hypothesis="test", depth=0,
            evidence_for=[ep_for], evidence_against=[ep_against],
        )
        # With default (inf) half-life, should use standard ratio
        conf = engine._evaluate_confidence(node)
        w_ev = engine.calibrator.evidence_weight
        w_ch = engine.calibrator.child_weight
        expected = node.evidence_ratio * w_ev + 0.5 * w_ch
        assert conf == pytest.approx(expected)

    def test_reasoning_with_default_decay_produces_valid_plan(self):
        """Full reasoning session with default decay should work."""
        engine = RecursiveReasoningEngine()
        plan = engine.reason("simple task")
        assert plan is not None
        assert len(plan.plans) > 0

    def test_reasoning_with_explicit_decay_produces_valid_plan(self):
        """Full reasoning session with explicit decay should work."""
        engine = RecursiveReasoningEngine(evidential_half_life=3600.0)
        plan = engine.reason("simple task")
        assert plan is not None
        assert len(plan.plans) > 0


# ═══════════════════════════════════════════════════════════════════
# Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases for evidential decay."""

    def test_zero_weight_evidence(self):
        """Zero-weight evidence should remain zero after decay."""
        decayed = evidential_decay(0.0, 1000.0, now=5000.0, half_life=3600.0)
        assert decayed == 0.0

    def test_very_large_half_life_approximates_no_decay(self):
        """Very large half-life should produce negligible decay."""
        decayed = evidential_decay(1.0, 1000.0, now=2000.0, half_life=1e15)
        assert abs(1.0 - decayed) < 1e-10

    def test_negative_weight_clamped_by_min(self):
        """Decay result should never be negative (min_weight >= 0)."""
        # This tests that the math never produces negative results
        for age in [0, 100, 3600, 36000, 360000]:
            decayed = evidential_decay(
                1.0, 1000.0, now=1000.0 + age, half_life=3600.0
            )
            assert decayed >= 0.0

    def test_mixed_fresh_and_stale_evidence(self):
        """Node with mixed fresh/stale evidence should weigh fresh more."""
        fresh = _ep(supports=True, weight=0.6, created_at=3999.0)
        stale = _ep(supports=True, weight=1.0, created_at=1000.0)
        node = ReasoningNode(
            hypothesis="test", depth=0, evidence_for=[fresh, stale],
        )
        now = 4000.0
        ratio_at = node.evidence_ratio_at(now=now, half_life=3600.0)
        # Standard ratio: (0.6 + 1.0) / (0.6 + 1.0) = 1.0
        # Decayed: fresh ≈ 0.6, stale ≈ 0.5 → total ≈ 1.1
        # Both are for, so ratio is still 1.0
        assert ratio_at == pytest.approx(1.0)

        # But add an against point to make the ratio meaningful
        fresh_against = _ep(supports=False, weight=0.8, created_at=3999.0)
        node2 = ReasoningNode(
            hypothesis="test", depth=0,
            evidence_for=[stale], evidence_against=[fresh_against],
        )
        ratio_at2 = node2.evidence_ratio_at(now=now, half_life=3600.0)
        standard = node2.evidence_ratio
        # Stale for evidence is decayed → ratio drops
        assert ratio_at2 < standard

    def test_evidence_point_repr(self):
        """EvidencePoint repr should not crash with created_at."""
        ep = _ep(created_at=1700000000.0)
        r = repr(ep)
        assert "EvidencePoint" in r
