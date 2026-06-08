"""LLM Steering — user commands → VLA prompts via an LLM + quadcamera context.

The high-level idea: a human types natural-language commands
(``"reach the red cube"``, ``"back away slowly"``).  The steering
module packages the current multi-camera observation plus the user
command, sends it to an LLM, and receives back a concise VLA prompt.

This closes the outer loop: human → LLM → VLA → MuJoCo.

For demos without API keys a ``MockLLMSteering`` variant is provided.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from isonome.llm.client import LLMClient

logger = logging.getLogger("isonome.sim.llm_steering")


class LLMSteering:
    """Production steering: real LLM calls to generate VLA prompts.

    Parameters
    ----------
    client:
        Configured :class:`LLMClient` (OpenAI or Anthropic).
    system_prompt:
        System message that describes the robot and camera setup.
    """

    _DEFAULT_SYSTEM: str = (
        "You are the high-level planner for a 7-DOF robot arm.\n"
        "The robot receives vision from 4 cameras: front, right, back, top.\n"
        "Your job is to translate the user's natural-language command into a\n"
        "short, precise VLA (Vision-Language-Action) prompt that the low-level\n"
        "policy can execute.  Keep responses to one sentence.\n"
        "Example: 'User: reach the red cube' → 'reach forward and grasp the red cube'"
    )

    def __init__(
        self,
        client: LLMClient | None = None,
        system_prompt: str = "",
    ) -> None:
        self._client = client
        self._system = system_prompt or self._DEFAULT_SYSTEM
        self._history: list[dict[str, str]] = []

    async def generate_intent(
        self,
        user_command: str,
        observation: dict[str, Any],
    ) -> str:
        """Return a VLA intent string for *user_command*.

        *observation* is the dict returned by
        :meth:`MuJoCoBridge.get_observation`.
        """
        if self._client is None:
            logger.warning("No LLM client configured; echoing user command")
            return user_command

        n_cameras = 1
        images = observation.get("image")
        if isinstance(images, list):
            n_cameras = len(images)

        prompt = (
            f"User command: {user_command}\n"
            f"Current observation: {n_cameras} camera(s), "
            f"proprioception shape {list(observation.get('proprioception', np.array([])).shape)}\n"
            f"Generate the VLA prompt:"
        )

        try:
            intent = await self._client.complete(
                prompt=prompt,
                system=self._system,
                max_tokens=128,
            )
            intent = intent.strip().strip('"').strip("'")
            self._history.append({"user": user_command, "intent": intent})
            logger.info("LLM steering", extra={"user": user_command, "intent": intent})
            return intent
        except Exception as exc:
            logger.error("LLM steering failed", extra={"error": str(exc)})
            return user_command


class MockLLMSteering:
    """Deterministic steering for demos without API keys.

    Maps a handful of known commands to canonical intents; everything
    else is echoed back with a ``'reach the '`` prefix if it looks like
    a target name.
    """

    _MAP: dict[str, str] = {
        "reach": "reach the target",
        "reach the target": "reach the target",
        "grab": "grasp the nearest object",
        "home": "return to home pose",
        "stop": "hold position",
        "up": "reach upward",
        "down": "reach downward",
        "left": "reach to the left",
        "right": "reach to the right",
    }

    def __init__(self) -> None:
        self._history: list[dict[str, str]] = []

    async def generate_intent(
        self,
        user_command: str,
        observation: dict[str, Any] | None = None,
    ) -> str:
        cmd = user_command.strip().lower()
        intent = self._MAP.get(cmd, "")
        if not intent:
            # Fallback: if it mentions a color or object, assume reach
            intent = f"reach the {user_command.strip()}"
        self._history.append({"user": user_command, "intent": intent})
        logger.info("Mock LLM steering", extra={"user": user_command, "intent": intent})
        return intent
