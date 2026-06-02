from __future__ import annotations

import asyncio
import logging
import time
from isonome.core.state import MotorCommand
from isonome.utils.logging import get_layer_logger


class ReflexWatchdog:
    """Hard kill switch for Reflex layer.

    Monitors Reflex tick latency. If Reflex misses its deadline,
    triggers emergency stop (zeros all motors, bypasses all layers).
    """

    def __init__(self, max_latency_ms: float = 10.0) -> None:
        self._max_latency_ms = max_latency_ms
        self._last_reflex_time = 0.0
        self._logger = get_layer_logger("safety.watchdog")
        self._emergency = False

    def record_tick(self) -> None:
        self._last_reflex_time = time.monotonic()

    def check(self) -> bool:
        """Returns True if Reflex is healthy, False if deadline missed."""
        if self._last_reflex_time == 0.0:
            return True
        latency_ms = (time.monotonic() - self._last_reflex_time) * 1000
        if latency_ms > self._max_latency_ms:
            self._logger.critical(
                "watchdog_deadline_missed",
                extra={"latency_ms": latency_ms},
            )
            return False
        return True

    def emergency_stop(self) -> MotorCommand:
        """Bypasses all layers -- zeros all motors."""
        self._emergency = True
        self._logger.critical("watchdog_emergency_stop")
        return MotorCommand(emergency_stop=True)

    @property
    def is_emergency(self) -> bool:
        return self._emergency
