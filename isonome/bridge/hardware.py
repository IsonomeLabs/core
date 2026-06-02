from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from isonome.core.state import SensorState, MotorCommand


class HardwareBridge(ABC):
    """Abstract hardware bridge -- ROS2 / MQTT / Serial.

    Concrete implementations left for the user.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to hardware."""
        ...

    @abstractmethod
    async def read_sensors(self) -> SensorState:
        """Read the current sensor state from hardware."""
        ...

    @abstractmethod
    async def write_motors(self, cmd: MotorCommand) -> None:
        """Send motor commands to hardware."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Release hardware resources."""
        ...


class StubHardwareBridge(HardwareBridge):
    """No-op bridge for testing."""

    async def connect(self) -> None:
        logging.getLogger("isonome.bridge.hardware").info("stub_connect")

    async def read_sensors(self) -> SensorState:
        return SensorState()

    async def write_motors(self, cmd: MotorCommand) -> None:
        pass

    async def disconnect(self) -> None:
        pass
