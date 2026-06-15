"""Domain randomization for URDF parameters and lighting.

Architecture gap #3, Step 1: ``Domain Randomization`` over mass, friction,
damping, and lighting.  The open-source implementation operates directly on
URDF XML and returns a dictionary of sim-lighting overrides so it can be used
with any backend that accepts URDF files and lighting parameters.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from isonome.praxis.calibration.config import DomainRandomizationConfig
from isonome.utils.logging import get_layer_logger


class DomainRandomizer:
    """Apply domain randomization to a URDF in memory.

    Parameters
    ----------
    config:
        Randomization ranges.  ``None`` disables a particular axis.
    """

    def __init__(self, config: DomainRandomizationConfig | None = None) -> None:
        self._config = config or DomainRandomizationConfig()
        self._logger = get_layer_logger("praxis.calibration.domain_randomization")

    def randomize(
        self,
        urdf_path: str | Path,
        *,
        seed: int | None = None,
    ) -> tuple[ET.Element, dict[str, Any]]:
        """Return a randomized URDF root and a lighting override dict.

        Parameters
        ----------
        urdf_path:
            Source URDF path.
        seed:
            Override RNG seed.  Defaults to ``config.seed``.

        Returns
        -------
        ``(root, lighting)`` where ``root`` is the randomized URDF element and
        ``lighting`` is a dict with ``intensity`` and ``seed``.
        """
        seed = seed if seed is not None else self._config.seed
        rng = random.Random(seed)

        tree = ET.parse(Path(urdf_path))
        root = copy.deepcopy(tree.getroot())

        for link in root.findall("link"):
            self._randomize_link(link, rng)

        for joint in root.findall("joint"):
            self._randomize_joint(joint, rng)

        lighting: dict[str, Any] = {}
        if self._config.lighting_intensity_range is not None:
            lo, hi = self._config.lighting_intensity_range
            lighting["intensity"] = rng.uniform(lo, hi)
            lighting["seed"] = seed

        self._logger.info(
            "domain_randomize",
            extra={
                "urdf": str(urdf_path),
                "seed": seed,
                "lighting": lighting,
            },
        )
        return root, lighting

    def randomize_to_file(
        self,
        urdf_path: str | Path,
        output_path: str | Path,
        *,
        seed: int | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        """Randomize a URDF and write it to ``output_path``."""
        root, lighting = self.randomize(urdf_path, seed=seed)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
        return output_path, lighting

    def _randomize_link(self, link: ET.Element, rng: random.Random) -> None:
        cfg = self._config
        inertial = link.find("inertial")
        if inertial is not None and cfg.mass_scale_range is not None:
            mass_elem = inertial.find("mass")
            if mass_elem is not None:
                original = float(mass_elem.get("value", "1.0"))
                scale = rng.uniform(*cfg.mass_scale_range)
                mass_elem.set("value", f"{original * scale:.6f}")

        for collision in link.findall("collision"):
            self._randomize_friction(collision, rng)

    def _randomize_joint(self, joint: ET.Element, rng: random.Random) -> None:
        cfg = self._config
        dynamics = joint.find("dynamics")
        if dynamics is not None and cfg.damping_scale_range is not None:
            damping = dynamics.get("damping")
            if damping is not None:
                original = float(damping)
                scale = rng.uniform(*cfg.damping_scale_range)
                dynamics.set("damping", f"{original * scale:.6f}")

    def _randomize_friction(self, collision: ET.Element, rng: random.Random) -> None:
        cfg = self._config
        if cfg.friction_scale_range is None:
            return

        friction = collision.find("surface/material/contact/friction_coefficient")
        if friction is not None:
            original = float(friction.get("value", "1.0"))
            scale = rng.uniform(*cfg.friction_scale_range)
            friction.set("value", f"{original * scale:.6f}")
            return

        # Fallback: look for ODE / MuJoCo friction attributes.
        surface = collision.find("surface")
        if surface is not None:
            ode = surface.find("friction/ode")
            if ode is not None:
                mu = ode.get("mu")
                if mu is not None:
                    original = float(mu)
                    scale = rng.uniform(*cfg.friction_scale_range)
                    ode.set("mu", f"{original * scale:.6f}")
