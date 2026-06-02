"""Isonome v0.2 state models — Frozen Brain + Learned Nervous System.

All tensor-bearing models use custom validators for torch.Tensor and override
model_dump() to serialize tensors as nested lists.
"""
from __future__ import annotations

import time
import warnings
from enum import Enum
from typing import Any, List, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# v0.2 State Models
# ---------------------------------------------------------------------------

class RawSensorState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Uncorrected proprioception and camera frames. Never post-processed.

    This is RAW sensor data. The brain (VLA policy) receives RAW state only.
    Never feed post-kernel corrected states into the perception pipeline.
    """

    proprioception: torch.Tensor
    camera_frames: List[torch.Tensor] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)

    @field_validator("proprioception", mode="before")
    @classmethod
    def _validate_tensor(cls, v: Any) -> torch.Tensor:
        return torch.as_tensor(v)

    @field_validator("camera_frames", mode="before")
    @classmethod
    def _validate_tensor_list(cls, v: Any) -> List[torch.Tensor]:
        if v is None:
            return []
        return [torch.as_tensor(item) for item in v]

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        d["proprioception"] = self.proprioception.tolist()
        d["camera_frames"] = [f.tolist() for f in self.camera_frames]
        return d


class CanonicalActionChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """VLA-native action space. Body-agnostic.

    Output of the frozen VLA policy (e.g., π0.7). This is NOT motor commands
    for the specific robot — it lives in a canonical action space (typically
    14-DOF end-effector intent).
    """

    actions: torch.Tensor  # shape: [chunk_size, canonical_dim]
    is_frozen_policy_output: Literal[True] = True

    @field_validator("actions", mode="before")
    @classmethod
    def _validate_tensor(cls, v: Any) -> torch.Tensor:
        return torch.as_tensor(v)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        d["actions"] = self.actions.tolist()
        return d


class CorrectedMotorCommand(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Robot-specific motor commands after kernel correction."""

    commands: torch.Tensor
    robot_hash: str = ""

    @field_validator("commands", mode="before")
    @classmethod
    def _validate_tensor(cls, v: Any) -> torch.Tensor:
        return torch.as_tensor(v)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        d["commands"] = self.commands.tolist()
        return d


class CortexAdvice(BaseModel):
    """Natural language advice from Cortex to JEPA."""

    text: str
    priority: int = 1  # Higher = prepend to prompt first


class ExecutionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """What actually happened after motor execution."""

    final_proprioception: torch.Tensor
    success: bool = True
    error_metric: float = 0.0

    @field_validator("final_proprioception", mode="before")
    @classmethod
    def _validate_tensor(cls, v: Any) -> torch.Tensor:
        return torch.as_tensor(v)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        d["final_proprioception"] = self.final_proprioception.tolist()
        return d


class Discrepancy(BaseModel):
    """A single observed difference between intent and outcome."""

    intended: CanonicalActionChunk
    actual: ExecutionResult
    raw_state: RawSensorState
    timestamp: float = Field(default_factory=time.time)


class JointLimits(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Per-joint safety limits."""

    lower: torch.Tensor
    upper: torch.Tensor

    @field_validator("lower", "upper", mode="before")
    @classmethod
    def _validate_tensor(cls, v: Any) -> torch.Tensor:
        return torch.as_tensor(v)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        d["lower"] = self.lower.tolist()
        d["upper"] = self.upper.tolist()
        return d


class MotorCommandChunk(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """A chunk of motor commands at policy frequency."""

    commands: torch.Tensor  # shape: [chunk_size, robot_dof]

    @field_validator("commands", mode="before")
    @classmethod
    def _validate_tensor(cls, v: Any) -> torch.Tensor:
        return torch.as_tensor(v)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        d["commands"] = self.commands.tolist()
        return d


class MotorCommand(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Single-step motor command at control frequency."""

    command: torch.Tensor

    @field_validator("command", mode="before")
    @classmethod
    def _validate_tensor(cls, v: Any) -> torch.Tensor:
        return torch.as_tensor(v)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        d["command"] = self.command.tolist()
        return d


class SafeMotorCommand(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Motor command after safety enforcement."""

    command: torch.Tensor
    was_clamped: bool = False
    emergency_stop: bool = False

    @field_validator("command", mode="before")
    @classmethod
    def _validate_tensor(cls, v: Any) -> torch.Tensor:
        return torch.as_tensor(v)

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        d = super().model_dump(**kwargs)
        d["command"] = self.command.tolist()
        return d


# ---------------------------------------------------------------------------
# v0.1 Legacy Models (deprecated, kept for backward compatibility)
# ---------------------------------------------------------------------------

class JointReading(BaseModel):
    name: str
    position: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0


class ContactReading(BaseModel):
    link_name: str
    force: float = 0.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)


class IMUReading(BaseModel):
    linear_acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


class SensorState(BaseModel):
    """Deprecated v0.1 sensor state. Use RawSensorState."""

    timestamp: float = Field(default_factory=time.time)
    joints: list[JointReading] = Field(default_factory=list)
    contacts: list[ContactReading] = Field(default_factory=list)
    imu: IMUReading = Field(default_factory=IMUReading)
    camera: dict[str, Any] = Field(default_factory=dict)
    extras: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any):
        warnings.warn(
            "SensorState is deprecated in v0.2. Use RawSensorState.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)


class Adjustment(BaseModel):
    """Deprecated v0.1 JEPA modulation."""

    position_deltas: dict[str, float] = Field(default_factory=dict)
    velocity_deltas: dict[str, float] = Field(default_factory=dict)
    effort_deltas: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PredictedState(BaseModel):
    timestamp: float = 0.0
    joint_positions: dict[str, float] = Field(default_factory=dict)
    joint_velocities: dict[str, float] = Field(default_factory=dict)
    contacts: list[ContactReading] = Field(default_factory=list)


class WorldModel(BaseModel):
    """Deprecated v0.1 JEPA world model."""

    current_state: PredictedState = Field(default_factory=PredictedState)
    predicted_states: list[PredictedState] = Field(default_factory=list)
    confidence: float = 1.0
    anomaly_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatchType(str, Enum):
    HYPERPARAMETER = "hyperparameter"
    CODE = "code"
    BEHAVIOR_TREE = "behavior_tree"
    CONFIG = "config"


class Patch(BaseModel):
    """Deprecated v0.1 patch."""

    patch_id: str = ""
    patch_type: PatchType = PatchType.HYPERPARAMETER
    target_layer: str = ""
    description: str = ""
    changes: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    proposer: str = ""


class ErrorEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    error_class: str = ""
    message: str = ""
    layer: str = ""
    severity: str = "warning"
    context: dict[str, Any] = Field(default_factory=dict)
