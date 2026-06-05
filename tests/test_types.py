"""Tests for isonome.types — core type system.

Covers: fundamental types, pillars, tension system, agent identity/lifecycle,
signal/feedback, task representation, exceptions, and pillar protocol.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
import pytest
from pydantic import ValidationError

from isonome.types import (
    AgentID,
    AgentIdentity,
    AgentLifecycle,
    AgentState,
    CognitionError,
    EquilibriumError,
    Feedback,
    IsonomeError,
    MnemeError,
    Pillar,
    PillarError,
    PillarProtocol,
    PraxisError,
    Signal,
    Task,
    TaskComplexity,
    TaskStatus,
    TensionAxis,
    TensionID,
    TensionOscillationError,
    TensionSnapshot,
    Timestamp,
    Token,
    now,
)


# ═══════════════════════════════════════════════════════════
# Fundamental types
# ═══════════════════════════════════════════════════════════

class TestFundamentalTypes:
    def test_now_returns_utc_datetime(self):
        t = now()
        assert isinstance(t, datetime)
        assert t.tzinfo is not None

    def test_now_is_recent(self):
        t = now()
        delta = (datetime.now(timezone.utc) - t).total_seconds()
        assert abs(delta) < 2.0

    def test_agent_id_is_uuid(self):
        aid = uuid4()
        assert isinstance(aid, AgentID)

    def test_tension_id_is_str(self):
        tid: TensionID = "explore_exploit"
        assert isinstance(tid, str)

    def test_token_is_int(self):
        t: Token = 42
        assert isinstance(t, int)

    def test_timestamp_is_datetime(self):
        ts: Timestamp = now()
        assert isinstance(ts, datetime)


# ═══════════════════════════════════════════════════════════
# Pillar enum
# ═══════════════════════════════════════════════════════════

class TestPillar:
    def test_three_pillars(self):
        assert set(Pillar) == {Pillar.COGNITION, Pillar.PRAXIS, Pillar.MNEME}

    def test_pillar_values(self):
        assert Pillar.COGNITION.value == "cognition"
        assert Pillar.PRAXIS.value == "praxis"
        assert Pillar.MNEME.value == "mneme"

    def test_pillar_is_string_enum(self):
        assert isinstance(Pillar.COGNITION, str)
        assert Pillar.COGNITION == "cognition"

    def test_pillar_from_value(self):
        assert Pillar("cognition") is Pillar.COGNITION
        assert Pillar("praxis") is Pillar.PRAXIS
        assert Pillar("mneme") is Pillar.MNEME


# ═══════════════════════════════════════════════════════════
# TensionAxis
# ═══════════════════════════════════════════════════════════

class TestTensionAxis:
    def test_default_construction(self):
        axis = TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="left", pole_right="right")
        assert axis.position == 0.0
        assert axis.default_position == 0.0
        assert axis.damping == 0.3
        assert axis.learning_rate == 0.1
        assert axis.clip == (-1.0, 1.0)

    def test_custom_construction(self):
        axis = TensionAxis(
            id="explore_exploit",
            pillar=Pillar.COGNITION,
            pole_left="explore",
            pole_right="exploit",
            position=0.5,
            default_position=0.1,
            damping=0.5,
            learning_rate=0.2,
        )
        assert axis.position == 0.5
        assert axis.damping == 0.5

    def test_frozen_immutability(self):
        axis = TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r")
        with pytest.raises(ValidationError):
            axis.position = 0.5

    def test_position_bounds(self):
        with pytest.raises(ValidationError):
            TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", position=1.5)
        with pytest.raises(ValidationError):
            TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", position=-1.5)

    def test_damping_bounds(self):
        with pytest.raises(ValidationError):
            TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", damping=1.5)
        with pytest.raises(ValidationError):
            TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", damping=-0.1)

    def test_adjust_basic(self):
        axis = TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", damping=0.0)
        new_axis = axis.adjust(0.3)
        assert new_axis.position == pytest.approx(0.3)
        # Original is unchanged (immutable)
        assert axis.position == 0.0

    def test_adjust_with_damping(self):
        axis = TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", damping=0.5)
        new_axis = axis.adjust(1.0)
        # effective_delta = 1.0 * (1.0 - 0.5) = 0.5
        assert new_axis.position == pytest.approx(0.5)

    def test_adjust_clipping(self):
        axis = TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", damping=0.0)
        new_axis = axis.adjust(5.0)
        assert new_axis.position == 1.0  # Clipped to upper bound

    def test_adjust_no_clip(self):
        axis = TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", damping=0.0)
        new_axis = axis.adjust(5.0, clip=False)
        assert new_axis.position == 5.0  # Not clipped

    def test_adjust_negative(self):
        axis = TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", damping=0.0)
        new_axis = axis.adjust(-0.7)
        assert new_axis.position == pytest.approx(-0.7)

    def test_distance_from_default(self):
        axis = TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", position=0.3, default_position=0.1)
        assert axis.distance_from_default() == pytest.approx(0.2)

    def test_distance_from_default_at_home(self):
        axis = TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", position=0.0, default_position=0.0)
        assert axis.distance_from_default() == 0.0

    def test_repr(self):
        axis = TensionAxis(id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", position=0.5)
        r = repr(axis)
        assert "test" in r
        assert "+0.500" in r

    def test_custom_clip_bounds(self):
        axis = TensionAxis(
            id="test", pillar=Pillar.COGNITION, pole_left="l", pole_right="r",
            clip=(-0.5, 0.5), damping=0.0,
        )
        new_axis = axis.adjust(1.0)
        assert new_axis.position == 0.5  # Clipped to custom upper bound


# ═══════════════════════════════════════════════════════════
# TensionSnapshot
# ═══════════════════════════════════════════════════════════

class TestTensionSnapshot:
    def test_basic_construction(self):
        axes = frozenset({
            TensionAxis(id="a", pillar=Pillar.COGNITION, pole_left="l", pole_right="r"),
            TensionAxis(id="b", pillar=Pillar.PRAXIS, pole_left="l", pole_right="r"),
        })
        snap = TensionSnapshot(axes=axes)
        assert len(snap.axes) == 2

    def test_get_axis_by_id(self):
        axis_a = TensionAxis(id="a", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", position=0.3)
        axis_b = TensionAxis(id="b", pillar=Pillar.PRAXIS, pole_left="l", pole_right="r", position=-0.2)
        snap = TensionSnapshot(axes=frozenset({axis_a, axis_b}))
        assert snap.get("a").position == pytest.approx(0.3)
        assert snap.get("b").position == pytest.approx(-0.2)

    def test_get_missing_axis(self):
        snap = TensionSnapshot(axes=frozenset())
        assert snap.get("nonexistent") is None

    def test_to_vector(self):
        axis_a = TensionAxis(id="a", pillar=Pillar.COGNITION, pole_left="l", pole_right="r", position=0.3)
        axis_b = TensionAxis(id="b", pillar=Pillar.PRAXIS, pole_left="l", pole_right="r", position=-0.2)
        snap = TensionSnapshot(axes=frozenset({axis_a, axis_b}))
        vec = snap.to_vector(["a", "b"])
        assert vec.shape == (2,)
        assert vec[0] == pytest.approx(0.3)
        assert vec[1] == pytest.approx(-0.2)

    def test_to_vector_missing_axis_defaults_zero(self):
        snap = TensionSnapshot(axes=frozenset())
        vec = snap.to_vector(["a", "b"])
        assert vec.shape == (2,)
        np.testing.assert_array_equal(vec, [0.0, 0.0])

    def test_frozen(self):
        snap = TensionSnapshot(axes=frozenset())
        with pytest.raises(ValidationError):
            snap.trigger = "manual"

    def test_timestamp_auto_set(self):
        snap = TensionSnapshot(axes=frozenset())
        assert isinstance(snap.timestamp, datetime)


# ═══════════════════════════════════════════════════════════
# AgentIdentity & AgentLifecycle
# ═══════════════════════════════════════════════════════════

class TestAgentLifecycle:
    def test_all_lifecycle_states(self):
        expected = {"created", "bootstrapping", "idle", "reasoning", "acting", "consolidating", "paused", "terminated"}
        actual = {s.value for s in AgentLifecycle}
        assert actual == expected

    def test_lifecycle_is_string_enum(self):
        assert isinstance(AgentLifecycle.CREATED, str)


class TestAgentIdentity:
    def test_default_construction(self):
        identity = AgentIdentity(name="test-agent")
        assert identity.name == "test-agent"
        assert isinstance(identity.id, UUID)
        assert identity.version == "0.1.0"
        assert isinstance(identity.created_at, datetime)

    def test_name_required(self):
        with pytest.raises(ValidationError):
            AgentIdentity()

    def test_name_too_long(self):
        with pytest.raises(ValidationError):
            AgentIdentity(name="x" * 129)

    def test_name_empty(self):
        with pytest.raises(ValidationError):
            AgentIdentity(name="")

    def test_frozen(self):
        identity = AgentIdentity(name="test-agent")
        with pytest.raises(ValidationError):
            identity.name = "changed"

    def test_parent_id(self):
        parent = uuid4()
        identity = AgentIdentity(name="child", parent_id=parent)
        assert identity.parent_id == parent

    def test_parent_id_default_none(self):
        identity = AgentIdentity(name="orphan")
        assert identity.parent_id is None


class TestAgentState:
    def test_default_construction(self):
        identity = AgentIdentity(name="test")
        state = AgentState(identity=identity)
        assert state.lifecycle == AgentLifecycle.CREATED
        assert state.task_count == 0
        assert state.error_count == 0
        assert state.tokens_consumed == 0
        assert state.tensions is None

    def test_mutable(self):
        identity = AgentIdentity(name="test")
        state = AgentState(identity=identity)
        state.lifecycle = AgentLifecycle.IDLE
        assert state.lifecycle == AgentLifecycle.IDLE

    def test_negative_counts_rejected(self):
        identity = AgentIdentity(name="test")
        with pytest.raises(ValidationError):
            AgentState(identity=identity, task_count=-1)


# ═══════════════════════════════════════════════════════════
# Signal
# ═══════════════════════════════════════════════════════════

class TestSignal:
    def test_basic_construction(self):
        sig = Signal(source=Pillar.COGNITION, target=Pillar.PRAXIS, kind="plan_ready")
        assert sig.source == Pillar.COGNITION
        assert sig.target == Pillar.PRAXIS
        assert sig.kind == "plan_ready"
        assert sig.priority == 0
        assert isinstance(sig.id, UUID)

    def test_with_payload(self):
        sig = Signal(
            source=Pillar.PRAXIS, target=Pillar.MNEME,
            kind="store", payload={"content": "hello"},
        )
        assert sig.payload["content"] == "hello"

    def test_priority_bounds(self):
        with pytest.raises(ValidationError):
            Signal(source=Pillar.COGNITION, target=Pillar.PRAXIS, kind="test", priority=11)
        with pytest.raises(ValidationError):
            Signal(source=Pillar.COGNITION, target=Pillar.PRAXIS, kind="test", priority=-1)

    def test_frozen(self):
        sig = Signal(source=Pillar.COGNITION, target=Pillar.PRAXIS, kind="test")
        with pytest.raises(ValidationError):
            sig.kind = "changed"


# ═══════════════════════════════════════════════════════════
# Feedback
# ═══════════════════════════════════════════════════════════

class TestFeedback:
    def test_basic_construction(self):
        fb = Feedback(source=Pillar.COGNITION, tension_axis_id="explore_exploit", signal=0.5, reason="test")
        assert fb.signal == 0.5
        assert fb.confidence == 0.5
        assert fb.reason == "test"

    def test_signal_bounds(self):
        with pytest.raises(ValidationError):
            Feedback(source=Pillar.COGNITION, tension_axis_id="test", signal=1.5, reason="test")
        with pytest.raises(ValidationError):
            Feedback(source=Pillar.COGNITION, tension_axis_id="test", signal=-1.5, reason="test")

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Feedback(source=Pillar.COGNITION, tension_axis_id="test", signal=0.0, confidence=1.5, reason="test")
        with pytest.raises(ValidationError):
            Feedback(source=Pillar.COGNITION, tension_axis_id="test", signal=0.0, confidence=-0.1, reason="test")

    def test_reason_required(self):
        with pytest.raises(ValidationError):
            Feedback(source=Pillar.COGNITION, tension_axis_id="test", signal=0.0, reason="")

    def test_frozen(self):
        fb = Feedback(source=Pillar.COGNITION, tension_axis_id="test", signal=0.0, reason="test")
        with pytest.raises(ValidationError):
            fb.signal = 0.5


# ═══════════════════════════════════════════════════════════
# Task
# ═══════════════════════════════════════════════════════════

class TestTask:
    def test_basic_construction(self):
        task = Task(description="Do something")
        assert task.description == "Do something"
        assert task.complexity == TaskComplexity.SIMPLE
        assert task.status == TaskStatus.PENDING
        assert task.is_atomic()

    def test_with_subtasks(self):
        child1 = uuid4()
        child2 = uuid4()
        task = Task(description="Parent", subtasks=(child1, child2))
        assert not task.is_atomic()
        assert len(task.subtasks) == 2

    def test_complexity_levels(self):
        expected = {"trivial", "simple", "moderate", "complex", "wicked"}
        actual = {c.value for c in TaskComplexity}
        assert actual == expected

    def test_status_levels(self):
        expected = {"pending", "planning", "executing", "awaiting_feedback", "completed", "failed", "cancelled"}
        actual = {s.value for s in TaskStatus}
        assert actual == expected

    def test_frozen(self):
        task = Task(description="test")
        with pytest.raises(ValidationError):
            task.description = "changed"

    def test_description_required(self):
        with pytest.raises(ValidationError):
            Task()


# ═══════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════

class TestExceptions:
    def test_isonome_error_hierarchy(self):
        assert issubclass(EquilibriumError, IsonomeError)
        assert issubclass(TensionOscillationError, EquilibriumError)
        assert issubclass(PillarError, IsonomeError)
        assert issubclass(CognitionError, PillarError)
        assert issubclass(PraxisError, PillarError)
        assert issubclass(MnemeError, PillarError)

    def test_isonome_error_is_exception(self):
        assert issubclass(IsonomeError, Exception)

    def test_can_raise_and_catch(self):
        with pytest.raises(IsonomeError):
            raise CognitionError("test cognition error")

        with pytest.raises(PillarError):
            raise PraxisError("test praxis error")

        with pytest.raises(EquilibriumError):
            raise TensionOscillationError("oscillation detected")

    def test_exception_message(self):
        err = MnemeError("memory corruption")
        assert str(err) == "memory corruption"


# ═══════════════════════════════════════════════════════════
# PillarProtocol (structural check)
# ═══════════════════════════════════════════════════════════

class TestPillarProtocol:
    def test_protocol_has_required_methods(self):
        # Verify the protocol defines the expected interface
        assert hasattr(PillarProtocol, 'initialize')
        assert hasattr(PillarProtocol, 'receive_signal')
        assert hasattr(PillarProtocol, 'shutdown')

    def test_minimal_implementation(self):
        """A class satisfying the protocol should work."""
        class FakePillar:
            pillar = Pillar.COGNITION

            def initialize(self, agent_state: AgentState) -> None:
                pass

            def receive_signal(self, signal: Signal) -> None:
                pass

            def shutdown(self) -> None:
                pass

        fp = FakePillar()
        assert fp.pillar == Pillar.COGNITION
        fp.initialize(AgentState(identity=AgentIdentity(name="test")))
        fp.shutdown()
