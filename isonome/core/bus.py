from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Any, Callable, Awaitable


class Channel(str, Enum):
    SENSORS = "sensors"
    REFLEX_OUTPUT = "reflex_output"
    JEPA_ADJUSTMENT = "jepa_adjustment"
    CORTEX_ADVICE = "cortex_advice"
    PLASTICITY_PATCHES = "plasticity_patches"
    ERROR = "error"


class MessageBus:
    """Async inter-layer message bus with typed channels."""

    def __init__(self, maxsize: int = 100) -> None:
        self._queues: dict[Channel, asyncio.Queue] = {}
        self._subscribers: dict[Channel, list[Callable[[Any], Awaitable[None]]]] = {}
        self._running = False
        self._logger = logging.getLogger("isonome.bus")
        for ch in Channel:
            self._queues[ch] = asyncio.Queue(maxsize=maxsize)
            self._subscribers[ch] = []

    async def start(self) -> None:
        self._running = True
        self._logger.info("message_bus_started")

    async def stop(self) -> None:
        self._running = False
        self._logger.info("message_bus_stopped")

    async def publish(self, channel: Channel, message: Any) -> None:
        if not self._running:
            return
        try:
            self._queues[channel].put_nowait(message)
        except asyncio.QueueFull:
            self._logger.warning(
                "channel_full", extra={"channel": channel.value}
            )
            return
        for callback in self._subscribers[channel]:
            asyncio.create_task(callback(message))

    def subscribe(
        self, channel: Channel, callback: Callable[[Any], Awaitable[None]]
    ) -> None:
        self._subscribers[channel].append(callback)

    async def receive(
        self, channel: Channel, timeout: float | None = None
    ) -> Any:
        if timeout:
            return await asyncio.wait_for(
                self._queues[channel].get(), timeout=timeout
            )
        return await self._queues[channel].get()

    async def try_receive(
        self, channel: Channel
    ) -> tuple[bool, Any]:
        """Non-blocking receive. Returns (success, message)."""
        try:
            msg = self._queues[channel].get_nowait()
            return True, msg
        except asyncio.QueueEmpty:
            return False, None
