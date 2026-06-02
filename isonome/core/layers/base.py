from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from enum import Enum


class LayerState(str, Enum):
    IDLE = "idle"
    BOOTING = "booting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class LayerBase(ABC):
    """Abstract base for all four cognitive layers."""

    def __init__(self, name: str, frequency_hz: float = 10.0) -> None:
        self._name = name
        self._frequency_hz = frequency_hz
        self._state = LayerState.IDLE
        self._logger = logging.getLogger(f"isonome.layer.{name}")
        self._tick_count: int = 0
        self._last_tick_time: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def state(self) -> LayerState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._state == LayerState.RUNNING

    @property
    def tick_period(self) -> float:
        return 1.0 / self._frequency_hz if self._frequency_hz > 0 else float("inf")

    @property
    def tick_count(self) -> int:
        return self._tick_count

    async def boot(self) -> None:
        self._state = LayerState.BOOTING
        self._logger.info("booting", extra={"layer": self._name})
        try:
            await self.on_boot()
            self._state = LayerState.RUNNING
            self._logger.info("running", extra={"layer": self._name})
        except Exception as e:
            self._state = LayerState.ERROR
            self._logger.error(
                "boot_failed", extra={"layer": self._name, "error": str(e)}
            )
            raise

    async def tick(self) -> None:
        if self._state != LayerState.RUNNING:
            return
        now = time.monotonic()
        try:
            await self.on_tick()
            self._tick_count += 1
            self._last_tick_time = now
        except Exception as e:
            self._logger.error(
                "tick_failed",
                extra={"layer": self._name, "error": str(e), "tick": self._tick_count},
            )
            raise

    async def shutdown(self) -> None:
        self._state = LayerState.STOPPING
        self._logger.info("shutting_down", extra={"layer": self._name})
        try:
            await self.on_shutdown()
        except Exception as e:
            self._logger.error(
                "shutdown_failed", extra={"layer": self._name, "error": str(e)}
            )
        finally:
            self._state = LayerState.IDLE

    async def adapt(self) -> None:
        if self._state != LayerState.RUNNING:
            return
        try:
            await self.on_adapt()
        except Exception as e:
            self._logger.error(
                "adapt_failed", extra={"layer": self._name, "error": str(e)}
            )

    @abstractmethod
    async def on_boot(self) -> None: ...

    @abstractmethod
    async def on_tick(self) -> None: ...

    @abstractmethod
    async def on_shutdown(self) -> None: ...

    async def on_adapt(self) -> None:
        """Optional — override for Plasticity integration."""
        pass
