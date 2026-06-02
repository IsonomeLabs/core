from __future__ import annotations
from isonome.presets.built_in.pet import PetPreset
from isonome.presets.built_in.patrol import PatrolPreset

BUILT_IN_PRESETS: dict[str, type[PetPreset | PatrolPreset]] = {
    "pet": PetPreset,
    "patrol": PatrolPreset,
}
