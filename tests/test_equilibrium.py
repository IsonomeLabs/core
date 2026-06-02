"""Tests for the equilibrium engine."""

import pytest

from isonome.equilibrium import EquilibriumEngine
from isonome.types import Feedback, Pillar, TensionAxis


class TestTensionAxis:
    def test_adjust_positive_delta(self):
        axis = TensionAxis(
            id="test",
            pillar=Pillar.COGNITION,
            pole_left="left",
            pole_right="right",
            position=0.0,
            damping=0.0,
        )
        new_axis = axis.adjust(0.5)
        assert new_axis.position == 0.5
        # Original is frozen — unchanged
        assert axis.position == 0.0

    def test_adjust_with_damping(self):
        axis = TensionAxis(
            id="test",
            pillar=Pillar.COGNITION,
            pole_left="left",
            pole_right="right",
            position=0.0,
            damping=0.5,
        )
        new_axis = axis.adjust(1.0)
        assert new_axis.position == 0.5  # damped: 1.0 * (1 - 0.5) = 0.5

    def test_adjust_respects_clip(self):
        axis = TensionAxis(
            id="test",
            pillar=Pillar.COGNITION,
            pole_left="left",
            pole_right="right",
            position=0.9,
            damping=0.0,
            clip=(-1.0, 1.0),
        )
        new_axis = axis.adjust(0.5)
        assert new_axis.position == 1.0  # clipped at upper bound

    def test_distance_from_default(self):
        axis = TensionAxis(
            id="test",
            pillar=Pillar.COGNITION,
            pole_left="left",
            pole_right="right",
            position=0.7,
            default_position=0.2,
        )
        assert axis.distance_from_default() == pytest.approx(0.5)


class TestEquilibriumEngine:
    def test_initialization_with_default_axes(self):
        engine = EquilibriumEngine()
        assert len(engine.axes) == 8
        axis_ids = {a.id for a in engine.axes}
        assert "explore_exploit" in axis_ids
        assert "autonomy_safety" in axis_ids
        assert "consolidate_prune" in axis_ids

    def test_snapshot(self):
        engine = EquilibriumEngine()
        snap = engine.snapshot()
        assert len(snap.axes) == 8
        assert snap.agent_id is None

    def test_snapshot_with_agent_id(self):
        engine = EquilibriumEngine()
        from uuid import uuid4

        agent_id = uuid4()
        snap = engine.snapshot(agent_id=agent_id, trigger="test")
        assert snap.agent_id == agent_id
        assert snap.trigger == "test"

    def test_apply_feedback_moves_axis(self):
        engine = EquilibriumEngine()
        feedback = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.5,
            confidence=1.0,
            reason="test",
        )
        new_axis = engine.apply_feedback(feedback)
        assert new_axis.position != engine.DEFAULT_AXES[0].position
        assert engine.total_feedback_received == 1

    def test_apply_feedback_with_low_confidence(self):
        engine = EquilibriumEngine()
        axis_before = engine.get_axis("explore_exploit")
        assert axis_before is not None

        feedback = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=1.0,
            confidence=0.1,
            reason="low confidence test",
        )
        new_axis = engine.apply_feedback(feedback)
        # Low confidence → small movement
        movement = abs(new_axis.position - axis_before.position)
        assert movement < 0.5  # Should be small because damping + low confidence

    def test_apply_feedback_unknown_axis(self):
        engine = EquilibriumEngine()
        feedback = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="nonexistent",
            signal=0.5,
            confidence=1.0,
            reason="test",
        )
        with pytest.raises(KeyError, match="nonexistent"):
            engine.apply_feedback(feedback)

    def test_apply_feedback_batch_atomic(self):
        engine = EquilibriumEngine()
        feedbacks = [
            Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.3,
                confidence=1.0,
                reason="batch test 1",
            ),
            Feedback(
                source=Pillar.PRAXIS,
                tension_axis_id="autonomy_safety",
                signal=-0.2,
                confidence=1.0,
                reason="batch test 2",
            ),
        ]
        snapshot = engine.apply_feedback_batch(feedbacks)
        assert engine.total_feedback_received == 2

    def test_get_behavior_profile(self):
        engine = EquilibriumEngine()
        profile = engine.get_behavior_profile()
        assert "explore_exploit" in profile
        assert "autonomy_safety" in profile
        assert all(-1.0 <= v <= 1.0 for v in profile.values())

    def test_adjust_default_moves_set_point(self):
        engine = EquilibriumEngine()
        axis_before = engine.get_axis("explore_exploit")
        assert axis_before is not None
        old_default = axis_before.default_position

        new_axis = engine.adjust_default("explore_exploit", outcome_signal=0.5)
        assert new_axis.default_position != old_default

    def test_tension_distance_zero_at_start(self):
        engine = EquilibriumEngine()
        # All axes start at their default positions
        assert engine.tension_distance() == pytest.approx(0.0, abs=1e-6)

    def test_tension_distance_after_feedback(self):
        engine = EquilibriumEngine()
        feedback = Feedback(
            source=Pillar.COGNITION,
            tension_axis_id="explore_exploit",
            signal=0.8,
            confidence=1.0,
            reason="stress test",
        )
        engine.apply_feedback(feedback)
        assert engine.tension_distance() > 0.0

    def test_reset_returns_to_defaults(self):
        engine = EquilibriumEngine()
        # Move an axis
        engine.apply_feedback(
            Feedback(
                source=Pillar.COGNITION,
                tension_axis_id="explore_exploit",
                signal=0.9,
                confidence=1.0,
                reason="move then reset",
            )
        )
        assert engine.tension_distance() > 0.0

        engine.reset(keep_defaults=True)
        assert engine.tension_distance() == pytest.approx(0.0, abs=1e-6)

    def test_custom_axes(self):
        custom = [
            TensionAxis(
                id="custom_axis",
                pillar=Pillar.MNEME,
                pole_left="a",
                pole_right="b",
            )
        ]
        engine = EquilibriumEngine(axes=custom)
        assert len(engine.axes) == 1
        assert engine.axes[0].id == "custom_axis"

    def test_oscillation_detection_counts(self):
        """Rapid back-and-forth feedback should trigger oscillation counter."""
        # Use a custom axis with no damping so movement is maximally responsive
        from isonome.equilibrium import EquilibriumEngine
        from isonome.types import TensionAxis

        custom_axis = TensionAxis(
            id="osc_test",
            pillar=Pillar.COGNITION,
            pole_left="left",
            pole_right="right",
            damping=0.0,  # No damping — full responsiveness
            default_position=0.0,
        )
        engine = EquilibriumEngine(
            axes=[custom_axis],
            oscillation_window=4,
            oscillation_threshold=0.3,
        )

        # Send rapid alternating feedback with full confidence
        for i in range(8):
            sign = 1.0 if i % 2 == 0 else -1.0
            engine.apply_feedback(
                Feedback(
                    source=Pillar.COGNITION,
                    tension_axis_id="osc_test",
                    signal=sign,
                    confidence=1.0,
                    reason=f"oscillation test {i}",
                )
            )

        assert engine.total_oscillation_events > 0
