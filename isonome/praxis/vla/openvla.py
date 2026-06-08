"""OpenVLA wrapper — placeholder for the open-weights 7B VLA model.

When ``transformers`` is available the wrapper can be instantiated;
otherwise it raises on ``load`` with a helpful message.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from isonome.praxis.vla.base import VLABase

try:
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover
    HAS_TRANSFORMERS = False
    torch = None  # type: ignore[assignment]
    AutoModelForVision2Seq = None  # type: ignore[misc, assignment]
    AutoProcessor = None  # type: ignore[misc, assignment]


class OpenVLA(VLABase):
    """OpenVLA policy wrapper.

    Paper: *OpenVLA: An Open-Source Vision-Language-Action Model*
    (Kim et al., 2024).
    """

    def __init__(self, action_dim: int = 7, device: str = "auto") -> None:
        self._action_dim = action_dim
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        if HAS_TRANSFORMERS and torch is not None:
            self._dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        else:
            self._dtype = None

    # ------------------------------------------------------------------
    # VLABase interface
    # ------------------------------------------------------------------

    def load(self, checkpoint_path: str, **kwargs: Any) -> None:
        if not HAS_TRANSFORMERS:
            raise RuntimeError(
                "OpenVLA requires transformers + torch.  "
                "Install:  pip install transformers accelerate torch"
            )
        self._model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,
            torch_dtype=self._dtype,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
        )
        self._model.to(self._device)
        self._model.eval()

    def predict(self, obs: dict[str, Any]) -> np.ndarray:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        image = obs.get("image")
        intent = obs.get("intent", "")
        proprio = obs.get("proprioception")

        # Build prompt following OpenVLA convention
        prompt = f"In: What action should the robot take to {intent}?\nOut:"

        inputs = self._processor(prompt, image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(**inputs, max_new_tokens=128)
        text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # TODO: parse generated tokens into continuous action vector.
        # For now return zeros so the bridge does not crash.
        return np.zeros(self._action_dim, dtype=np.float32)
