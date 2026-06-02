from __future__ import annotations
from isonome.presets.base import Preset
from isonome.presets.built_in import BUILT_IN_PRESETS


def load_preset(name: str) -> Preset:
    """Load a preset by name. Raises ValueError if unknown."""
    if name not in BUILT_IN_PRESETS:
        available = ", ".join(sorted(BUILT_IN_PRESETS.keys()))
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return BUILT_IN_PRESETS[name]()
