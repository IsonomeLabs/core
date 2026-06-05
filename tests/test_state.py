"""Tests for isonome.core.state — v0.2 state models and v0.1 legacy models.

Covers tensor-bearing models (RawSensorState, CanonicalActionChunk, etc.),
CortexAdvice, ExecutionResult, Discrepancy, JointLimits, MotorCommand variants,
and legacy v0.1 models.
"""
from __future__ import annotations

import time
import warnings

import pytest
import torch
from pydantic import ValidationError

from isonome.core.state import (
    CanonicalActionChunk,
    ContactReading,
    CortexAdvice,
    CorrectedMotorCommand,
    Discrepancy,
    ErrorEvent,
    ExecutionResult,
    IMUReading,
    JointLimits,
    JointReading,
    MotorCommand,
    MotorCommandChunk,
    Patch,
    PatchType,
    PredictedState,
    RawSensorState,
    SafeMotorCommand,
    SensorState,
    WorldModel,
)


# ═══════════════════════════════════════════════════════════
# RawSensorState
# ═══════════════════════════════════════════════════════════

class TestRawSensorState:
    def test_basic_construction(self):
        state = RawSensorState(proprioception=torch.tensor([1.0, 2.0, 3.0]))
        assert state.proprioception.shape == (3,)
        assert state.camera_frames == []
        assert isinstance(state.timestamp, float)

    def test_from_list(self):
        state = RawSensorState(proprioception=[1.0, 2.0])
        assert isinstance(state.proprioception, torch.Tensor)
        assert state.proprioception.tolist() == [1.0, 2.0]

    def test_with_camera_frames(self):
        state = RawSensorState(
            proprioception=torch.zeros(3),
            camera_frames=[torch.zeros(64, 64, 3), torch.zeros(64, 64, 3)],
        )
        assert len(state.camera_frames) == 2

    def test_camera_frames_from_list(self):
        state = RawSensorState(
            proprioception=torch.zeros(3),
            camera_frames=[[[0.0]] * 64] * 64,
        )
        assert isinstance(state.camera_frames[0], torch.Tensor)

    def test_camera_frames_none_becomes_empty(self):
        state = RawSensorState(proprioception=torch.zeros(3), camera_frames=None)
        assert state.camera_frames == []

    def test_model_dump_serializes_tensors(self):
        state = RawSensorState(proprioception=torch.tensor([1.0, 2.0]))
        d = state.model_dump()
        assert isinstance(d["proprioception"], list)
        assert d["proprioception"] == [1.0, 2.0]
        assert isinstance(d["camera_frames"], list)


# ═══════════════════════════════════════════════════════════
# CanonicalActionChunk
# ═══════════════════════════════════════════════════════════

class TestCanonicalActionChunk:
    def test_basic_construction(self):
        chunk = CanonicalActionChunk(actions=torch.zeros(4, 14))
        assert chunk.actions.shape == (4, 14)
        assert chunk.is_frozen_policy_output is True

    def test_from_numpy_array(self):
        import numpy as np
        chunk = CanonicalActionChunk(actions=np.zeros((2, 7)))
        assert isinstance(chunk.actions, torch.Tensor)
        assert chunk.actions.shape == (2, 7)

    def test_model_dump(self):
        chunk = CanonicalActionChunk(actions=torch.tensor([[1.0, 2.0]]))
        d = chunk.model_dump()
        assert d["actions"] == [[1.0, 2.0]]


# ═══════════════════════════════════════════════════════════
# CorrectedMotorCommand
# ═══════════════════════════════════════════════════════════

class TestCorrectedMotorCommand:
    def test_basic_construction(self):
        cmd = CorrectedMotorCommand(commands=torch.zeros(6))
        assert cmd.commands.shape == (6,)
        assert cmd.robot_hash == ""

    def test_with_robot_hash(self):
        cmd = CorrectedMotorCommand(commands=torch.zeros(6), robot_hash="abc123")
        assert cmd.robot_hash == "abc123"

    def test_model_dump(self):
        cmd = CorrectedMotorCommand(commands=torch.tensor([1.0, 2.0]))
        d = cmd.model_dump()
        assert d["commands"] == [1.0, 2.0]


# ═══════════════════════════════════════════════════════════
# CortexAdvice
# ═══════════════════════════════════════════════════════════

class TestCortexAdvice:
    def test_basic_construction(self):
        advice = CortexAdvice(text="Reduce gain by 10%.")
        assert advice.text == "Reduce gain by 10%."
        assert advice.summary == ""
        assert advice.priority == "medium"
        assert advice.target_layer == "jepa"

    def test_with_all_fields(self):
        advice = CortexAdvice(
            text="Motor 0 overshot.",
            summary="Overshoot detected",
            priority="high",
            target_layer="jepa",
        )
        assert advice.priority == "high"
        assert advice.target_layer == "jepa"

    def test_priority_values(self):
        for p in ("low", "medium", "high", "critical"):
            advice = CortexAdvice(text="test", priority=p)
            assert advice.priority == p

    def test_target_layer_default_jepa(self):
        advice = CortexAdvice(text="test")
        assert advice.target_layer == "jepa"

    def test_custom_target_layer(self):
        advice = CortexAdvice(text="test", target_layer="reflex")
        assert advice.target_layer == "reflex"

    def test_invalid_priority_rejected(self):
        with pytest.raises(ValidationError):
            CortexAdvice(text="test", priority="urgent")

    def test_numeric_priority_rejected(self):
        with pytest.raises(ValidationError):
            CortexAdvice(text="test", priority=3)

    def test_model_dump(self):
        advice = CortexAdvice(
            text="Overshoot detected",
            summary="Motor 0 overshoot",
            priority="high",
        )
        d = advice.model_dump()
        assert d["text"] == "Overshoot detected"
        assert d["summary"] == "Motor 0 overshoot"
        assert d["priority"] == "high"
        assert d["target_layer"] == "jepa"


# ═══════════════════════════════════════════════════════════
# ExecutionResult

class TestExecutionResult:
    def test_basic_construction(self):
        result = ExecutionResult(final_proprioception=torch.zeros(7))
        assert result.success is True
        assert result.error_metric == 0.0

    def test_failure(self):
        result = ExecutionResult(
            final_proprioception=torch.zeros(7),
            success=False,
            error_metric=0.5,
        )
        assert result.success is False
        assert result.error_metric == 0.5

    def test_model_dump(self):
        result = ExecutionResult(final_proprioception=torch.tensor([1.0, 2.0]))
        d = result.model_dump()
        assert d["final_proprioception"] == [1.0, 2.0]


# ═══════════════════════════════════════════════════════════
# Discrepancy
# ═══════════════════════════════════════════════════════════

class TestDiscrepancy:
    def test_construction(self):
        disc = Discrepancy(
            intended=CanonicalActionChunk(actions=torch.zeros(1, 14)),
            actual=ExecutionResult(final_proprioception=torch.zeros(7)),
            raw_state=RawSensorState(proprioception=torch.zeros(3)),
        )
        assert isinstance(disc.intended, CanonicalActionChunk)
        assert isinstance(disc.actual, ExecutionResult)
        assert isinstance(disc.raw_state, RawSensorState)
        assert isinstance(disc.timestamp, float)


# ═══════════════════════════════════════════════════════════
# JointLimits
# ═══════════════════════════════════════════════════════════

class TestJointLimits:
    def test_construction(self):
        limits = JointLimits(lower=torch.tensor([-1.0, -2.0]), upper=torch.tensor([1.0, 2.0]))
        assert limits.lower.shape == (2,)
        assert limits.upper.shape == (2,)

    def test_from_lists(self):
        limits = JointLimits(lower=[-1.0, -2.0], upper=[1.0, 2.0])
        assert isinstance(limits.lower, torch.Tensor)

    def test_model_dump(self):
        limits = JointLimits(lower=torch.tensor([-1.0]), upper=torch.tensor([1.0]))
        d = limits.model_dump()
        assert d["lower"] == [-1.0]
        assert d["upper"] == [1.0]


# ═══════════════════════════════════════════════════════════
# MotorCommandChunk
# ═══════════════════════════════════════════════════════════

class TestMotorCommandChunk:
    def test_construction(self):
        chunk = MotorCommandChunk(commands=torch.zeros(4, 6))
        assert chunk.commands.shape == (4, 6)

    def test_model_dump(self):
        chunk = MotorCommandChunk(commands=torch.tensor([[1.0]]))
        d = chunk.model_dump()
        assert d["commands"] == [[1.0]]


# ═══════════════════════════════════════════════════════════
# MotorCommand (single step)
# ═══════════════════════════════════════════════════════════

class TestMotorCommand:
    def test_construction(self):
        cmd = MotorCommand(command=torch.zeros(6))
        assert cmd.command.shape == (6,)

    def test_model_dump(self):
        cmd = MotorCommand(command=torch.tensor([1.0, 2.0]))
        d = cmd.model_dump()
        assert d["command"] == [1.0, 2.0]


# ═══════════════════════════════════════════════════════════
# SafeMotorCommand
# ═══════════════════════════════════════════════════════════

class TestSafeMotorCommand:
    def test_default_construction(self):
        cmd = SafeMotorCommand(command=torch.zeros(6))
        assert cmd.was_clamped is False
        assert cmd.emergency_stop is False

    def test_clamped(self):
        cmd = SafeMotorCommand(command=torch.zeros(6), was_clamped=True)
        assert cmd.was_clamped is True

    def test_emergency_stop(self):
        cmd = SafeMotorCommand(command=torch.zeros(6), emergency_stop=True)
        assert cmd.emergency_stop is True

    def test_model_dump(self):
        cmd = SafeMotorCommand(command=torch.tensor([0.5]), was_clamped=True)
        d = cmd.model_dump()
        assert d["command"] == [0.5]
        assert d["was_clamped"] is True


# ═══════════════════════════════════════════════════════════
# Legacy v0.1 models
# ═══════════════════════════════════════════════════════════

class TestLegacyJointReading:
    def test_default_construction(self):
        j = JointReading(name="joint_0")
        assert j.position == 0.0
        assert j.velocity == 0.0
        assert j.effort == 0.0


class TestLegacyContactReading:
    def test_default_construction(self):
        c = ContactReading(link_name="link_0")
        assert c.force == 0.0
        assert c.position == (0.0, 0.0, 0.0)


class TestLegacyIMUReading:
    def test_default_construction(self):
        imu = IMUReading()
        assert imu.linear_acceleration == (0.0, 0.0, 0.0)
        assert imu.orientation == (0.0, 0.0, 0.0, 1.0)


class TestLegacySensorState:
    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            state = SensorState()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_default_construction(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            state = SensorState()
        assert state.joints == []
        assert state.contacts == []
        assert isinstance(state.imu, IMUReading)


class TestLegacyPatch:
    def test_default_construction(self):
        patch = Patch()
        assert patch.patch_type == PatchType.HYPERPARAMETER
        assert patch.confidence == 0.0

    def test_patch_types(self):
        expected = {"hyperparameter", "code", "behavior_tree", "config"}
        actual = {p.value for p in PatchType}
        assert actual == expected


class TestLegacyErrorEvent:
    def test_construction(self):
        event = ErrorEvent(error_class="ValueError", message="test error", layer="jepa")
        assert event.error_class == "ValueError"
        assert event.severity == "warning"


class TestLegacyWorldModel:
    def test_default_construction(self):
        wm = WorldModel()
        assert wm.confidence == 1.0
        assert wm.anomaly_score == 0.0
        assert isinstance(wm.current_state, PredictedState)
