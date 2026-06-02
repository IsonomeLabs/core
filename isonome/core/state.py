from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


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
    timestamp: float = Field(default_factory=time.time)
    joints: list[JointReading] = Field(default_factory=list)
    contacts: list[ContactReading] = Field(default_factory=list)
    imu: IMUReading = Field(default_factory=IMUReading)
    camera: dict[str, Any] = Field(default_factory=dict)
    extras: dict[str, Any] = Field(default_factory=dict)


class MotorCommand(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    joint_positions: dict[str, float] = Field(default_factory=dict)
    joint_velocities: dict[str, float] = Field(default_factory=dict)
    joint_efforts: dict[str, float] = Field(default_factory=dict)
    emergency_stop: bool = False


class Adjustment(BaseModel):
    """JEPA modulation of Reflex output — additive corrections."""

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
    """JEPA's internal state snapshot."""

    current_state: PredictedState = Field(default_factory=PredictedState)
    predicted_states: list[PredictedState] = Field(default_factory=list)
    confidence: float = 1.0
    anomaly_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CortexAdvice(BaseModel):
    """Natural-language + structured advice from Prefrontal Cortex. Never a MotorCommand."""

    summary: str = ""
    suggestions: list[str] = Field(default_factory=list)
    priority: str = "low"  # low, medium, high, critical
    target_layer: str = "jepa"  # only jepa allowed
    metadata: dict[str, Any] = Field(default_factory=dict)


class PatchType(str, Enum):
    HYPERPARAMETER = "hyperparameter"
    CODE = "code"
    BEHAVIOR_TREE = "behavior_tree"
    CONFIG = "config"


class Patch(BaseModel):
    """Code/config/hyperparameter change proposal from Plasticity."""

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
    severity: str = "warning"  # warning, error, critical
    context: dict[str, Any] = Field(default_factory=dict)
