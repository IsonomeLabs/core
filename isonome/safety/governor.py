from __future__ import annotations

import logging
import time
from collections import Counter
from enum import Enum
from isonome.core.config import SafetyConfig
from isonome.core.state import ErrorEvent, Patch
from isonome.utils.logging import get_layer_logger


class RobotState(str, Enum):
    OFFLINE = "offline"
    IDLE = "idle"
    BOOTING = "booting"
    RUNNING = "running"
    SAFE_STATIONARY = "safe_stationary"
    CRITICAL = "critical"


class SafetyGovernor:
    """Gates all plasticity events. Framework enforces safety rules.

    can_adapt() returns True only if:
    1. Robot state is OFFLINE or IDLE
    2. OR user flag permit_boot_adaptation=True during reboot (BOOTING)
    3. OR live trigger: same error class >= N times in window AND
       SAFE_STATIONARY
    """

    def __init__(
        self, config: SafetyConfig, agent: object | None = None
    ) -> None:
        self._config = config
        self._agent = agent
        self._logger = get_layer_logger("safety.governor")
        self._applied_patches: list[Patch] = []

    def can_adapt(self) -> bool:
        from isonome.core.agent import Agent

        agent = self._agent
        if not isinstance(agent, Agent):
            return False

        state = agent.state

        # Rule 1: Offline or idle -- always safe
        if state in (RobotState.OFFLINE, RobotState.IDLE):
            return True

        # Rule 2: Boot adaptation with explicit permission
        if state == RobotState.BOOTING and self._config.permit_boot_adaptation:
            self._logger.info("governor_boot_adaptation_permitted")
            return True

        # Rule 3: Live trigger -- repeated errors in safe state
        if state == RobotState.SAFE_STATIONARY:
            if self._check_repeated_errors(agent):
                self._logger.warning("governor_live_adaptation_triggered")
                return True

        return False

    def _check_repeated_errors(self, agent) -> bool:
        now = time.time()
        window = self._config.error_window_s
        threshold = self._config.error_repeat_threshold

        recent = [
            e for e in agent._error_buffer if now - e.timestamp < window
        ]
        if len(recent) < threshold:
            return False

        # Group by error class
        counts = Counter(e.error_class for e in recent)
        return any(c >= threshold for c in counts.values())

    async def apply_patches(self, patches: list[Patch]) -> None:
        """Transactional patch application: snapshot, apply, sim-validate, commit."""
        if not patches:
            return

        self._logger.info(
            "governor_applying_patches", extra={"count": len(patches)}
        )

        # TODO: Implement transactional snapshot + sim validation
        # For now, log patches but don't auto-apply
        for patch in patches:
            self._logger.info(
                "governor_patch_proposed",
                extra={
                    "patch_id": patch.patch_id,
                    "type": patch.patch_type.value,
                    "target": patch.target_layer,
                    "confidence": patch.confidence,
                },
            )
            self._applied_patches.append(patch)
