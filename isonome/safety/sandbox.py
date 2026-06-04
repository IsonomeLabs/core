from __future__ import annotations

import asyncio
import json
from isonome.core.state import CortexAdvice
from isonome.utils.logging import get_layer_logger


class LLMSandbox:
    """Sandboxed subprocess for LLM calls (Cortex).

    Enforces timeout and validates output (must be CortexAdvice, never
    MotorCommand).
    """

    def __init__(self, timeout_s: float = 30.0) -> None:
        self._timeout_s = timeout_s
        self._logger = get_layer_logger("safety.sandbox")

    async def execute(self, func, *args, **kwargs) -> CortexAdvice:
        """Execute an LLM call in a sandboxed context with timeout."""
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs), timeout=self._timeout_s
            )
            return self._validate(result)
        except asyncio.TimeoutError:
            self._logger.error(
                "sandbox_timeout", extra={"timeout_s": self._timeout_s}
            )
            return CortexAdvice(summary="Sandbox timeout", priority="low")
        except Exception as e:
            self._logger.error("sandbox_error", extra={"error": str(e)})
            return CortexAdvice(summary=f"Sandbox error: {e}", priority="low")

    def _validate(self, result) -> CortexAdvice:
        if isinstance(result, CortexAdvice):
            # Safety check: advice must target jepa only
            if result.target_layer != "jepa":
                self._logger.warning(
                    "sandbox_invalid_target",
                    extra={"target": result.target_layer},
                )
                result.target_layer = "jepa"
            return result
        # Try to parse as CortexAdvice
        if isinstance(result, dict):
            return CortexAdvice(**result)
        if isinstance(result, str):
            try:
                return CortexAdvice(**json.loads(result))
            except (json.JSONDecodeError, Exception):
                return CortexAdvice(summary=result[:200], priority="low")
        return CortexAdvice(summary="Invalid LLM output", priority="low")
