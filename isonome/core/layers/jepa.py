"""JEPA Layer — Frozen VLA Policy Loader.

The VLA policy (π0.7, SmolVLA, OpenVLA, etc.) is treated as a read-only,
frozen generalist brain. This layer loads the model and runs inference.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, List, Optional

import torch

from isonome.core.layers.base import LayerBase
from isonome.core.state import CanonicalActionChunk, CortexAdvice, RawSensorState
from isonome.utils.logging import get_layer_logger


class VLABackend(str, Enum):
    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    SMOLVLA = "smolvla"
    OPENVLA = "openvla"


def load_vla(backend: str, model_id: Optional[str] = None) -> Any:
    """Load a frozen VLA policy from HuggingFace.

    Args:
        backend: One of the VLABackend values.
        model_id: Optional HuggingFace model ID. If None, uses backend default.

    Returns:
        A model/policy object ready for inference.
    """
    backend = VLABackend(backend)
    if backend in (VLABackend.PI0, VLABackend.PI0_FAST):
        try:
            from lerobot import load_policy
        except ImportError as e:
            raise ImportError(
                "lerobot is required for pi0 backends. "
                "Install with: pip install isonome[pi0]"
            ) from e
        model_id = model_id or "physical-intelligence/fast"
        policy = load_policy(model_id)
        return policy

    elif backend == VLABackend.SMOLVLA:
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for SmolVLA. "
                "Install with: pip install isonome[vla]"
            ) from e
        model_id = model_id or "HuggingFaceTB/SmolVLA"
        model = AutoModelForVision2Seq.from_pretrained(model_id)
        processor = AutoProcessor.from_pretrained(model_id)
        return {"model": model, "processor": processor}

    elif backend == VLABackend.OPENVLA:
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for OpenVLA. "
                "Install with: pip install isonome[vla]"
            ) from e
        model_id = model_id or "openvla/openvla-7b"
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, trust_remote_code=True
        )
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        return {"model": model, "processor": processor}

    raise ValueError(f"Unknown VLA backend: {backend}")


class JEPALayer(LayerBase):
    """Frozen VLA brain — loads a pre-trained policy and runs inference.

    The policy is frozen: requires_grad=False on all parameters, and all
    forward passes run under torch.no_grad().
    """

    def __init__(
        self,
        frequency_hz: float = 10.0,
        backend: str = "openvla",
        model_id: Optional[str] = None,
    ) -> None:
        super().__init__(name="jepa", frequency_hz=frequency_hz)
        self._backend = backend
        self._model_id = model_id
        self._policy: Any = None
        self._logger = get_layer_logger("jepa")

    async def on_boot(self) -> None:
        self._logger.info(
            "jepa_layer_booting",
            extra={"backend": self._backend, "model_id": self._model_id},
        )
        try:
            self._policy = load_vla(self._backend, self._model_id)
        except ImportError as e:
            self._logger.warning(
                "jepa_policy_load_failed",
                extra={"error": str(e)},
            )
            self._policy = None
        if self._policy is not None and hasattr(self._policy, "parameters"):
            for param in self._policy.parameters():
                param.requires_grad = False
        self._logger.info("jepa_layer_ready")

    async def on_tick(self) -> None:
        pass  # tick logic driven externally by agent.py

    async def on_shutdown(self) -> None:
        self._logger.info("jepa_layer_shutdown")
        self._policy = None

    async def deliberate(
        self,
        raw_state: RawSensorState,
        prompt: str,
        advice_buffer: List[CortexAdvice],
    ) -> CanonicalActionChunk:
        """Generate a canonical action chunk from RAW sensor state.

        π0.7 receives RAW state only. Never feed post-kernel corrected states
        here or the system will oscillate.

        Args:
            raw_state: Uncorrected proprioception and camera frames. RAW data.
            prompt: Task description for the VLA.
            advice_buffer: Previous tick's advice from Cortex.

        Returns:
            CanonicalActionChunk in the VLA's native action space.
        """
        if self._policy is None:
            # Fallback: deterministic zero action for testing
            self._logger.warning("jepa_no_policy_fallback")
            return CanonicalActionChunk(actions=torch.zeros(1, 14))

        prompt = self._inject_advice(prompt, advice_buffer)

        with torch.no_grad():
            # Abstract inference — actual calling convention varies by backend.
            # For the open-source runtime, we support a generic interface.
            if isinstance(self._policy, dict):
                # SmolVLA / OpenVLA path
                actions = self._infer_hf(self._policy, raw_state, prompt)
            else:
                # pi0 / lerobot path
                actions = self._infer_lerobot(self._policy, raw_state, prompt)

        return CanonicalActionChunk(actions=actions)

    def _inject_advice(
        self, base_prompt: str, advice: List[CortexAdvice]
    ) -> str:
        """Prepend Cortex advice to the task prompt."""
        if not advice:
            return base_prompt
        advice_text = "\n".join(
            [a.text for a in sorted(advice, key=lambda x: x.priority, reverse=True)]
        )
        return (
            f"CORRECTIONS FROM PREVIOUS ATTEMPT:\n{advice_text}\n\n"
            f"TASK:\n{base_prompt}"
        )

    def _infer_lerobot(
        self, policy: Any, raw_state: RawSensorState, prompt: str
    ) -> torch.Tensor:
        # If the policy object has an infer method (e.g., MockVLABackend),
        # delegate to it. Otherwise return zeros as a placeholder.
        if hasattr(policy, "infer"):
            return policy.infer(raw_state, prompt)
        return torch.zeros(1, 14)

    def _infer_hf(
        self, policy: dict, raw_state: RawSensorState, prompt: str
    ) -> torch.Tensor:
        # Placeholder — actual transformers inference depends on processor
        # and model-specific calling conventions. Return a zero chunk for structure.
        model = policy.get("model")
        if hasattr(model, "infer"):
            return model.infer(raw_state, prompt)
        return torch.zeros(1, 14)
