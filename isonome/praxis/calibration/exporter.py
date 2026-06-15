"""Export a certified policy package as a ``.zip`` file.

Architecture gap #3, Step 4: ``Certified Policy Package (.zip)`` containing
manifest, per-agent policies, coordinator config, reflex gains, sim metrics,
certification video, and launcher.
"""
from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from isonome.utils.logging import get_layer_logger


@dataclass
class PolicyPackageArtifacts:
    """Inputs required to build a certified policy package."""

    manifest: dict[str, Any] = field(default_factory=dict)
    agent_configs: dict[str, Any] = field(default_factory=dict)
    coordinator_config: dict[str, Any] = field(default_factory=dict)
    reflex_gains: dict[str, Any] = field(default_factory=dict)
    sim_metrics: dict[str, Any] = field(default_factory=dict)
    policy_params: torch.Tensor | None = None
    certification_video_path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PolicyPackageExporter:
    """Create a ``.zip`` Certified Policy Package from calibration artifacts."""

    def __init__(self) -> None:
        self._logger = get_layer_logger("praxis.calibration.exporter")

    def export(
        self,
        artifacts: PolicyPackageArtifacts,
        output_path: str | Path,
    ) -> Path:
        """Write a ``.zip`` package to ``output_path``.

        The archive contains:

        * ``manifest.json``
        * ``agent_configs.json``
        * ``coordinator_config.json``
        * ``reflex_gains.json``
        * ``sim_metrics.json``
        * ``policy/policy.pt`` (optional torch state dict)
        * ``launcher.py``
        * ``certification_video.mp4`` (optional, copied from disk)
        * ``metadata.json``

        Returns
        -------
        Path to the created ``.zip`` file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            self._write_json(zf, "manifest.json", artifacts.manifest)
            self._write_json(zf, "agent_configs.json", artifacts.agent_configs)
            self._write_json(zf, "coordinator_config.json", artifacts.coordinator_config)
            self._write_json(zf, "reflex_gains.json", artifacts.reflex_gains)
            self._write_json(zf, "sim_metrics.json", artifacts.sim_metrics)
            self._write_json(zf, "metadata.json", artifacts.metadata)
            self._write_json(zf, "launcher.py", self._launcher_source())

            if artifacts.policy_params is not None:
                policy_buf = self._tensor_to_buffer(artifacts.policy_params)
                zf.writestr("policy/policy.pt", policy_buf)

            if artifacts.certification_video_path is not None:
                video_path = Path(artifacts.certification_video_path)
                if video_path.exists():
                    zf.write(video_path, arcname="certification_video.mp4")

        self._logger.info(
            "package_exported",
            extra={
                "path": str(output_path),
                "size_bytes": output_path.stat().st_size,
            },
        )
        return output_path

    @staticmethod
    def _write_json(zf: zipfile.ZipFile, arcname: str, data: Any) -> None:
        zf.writestr(arcname, json.dumps(data, indent=2, sort_keys=True, default=str))

    @staticmethod
    def _tensor_to_buffer(tensor: torch.Tensor) -> bytes:
        import io

        buf = io.BytesIO()
        torch.save({"policy_params": tensor}, buf)
        return buf.getvalue()

    @staticmethod
    def _launcher_source() -> str:
        return '''"""Certified policy package launcher.

This script is a minimal example of how to load the calibrated policy and run
it through the Isonome runtime.  Enterprise deployments usually replace this
with a ROS2 node or Isaac Sim orchestrator.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import torch

from isonome.core.app import IsonomeApp
from isonome.core.config import AppConfig


async def main() -> None:
    config = AppConfig.from_yaml(Path("config.yaml"))
    app = IsonomeApp(config)
    await app.run(duration_s=60.0)


if __name__ == "__main__":
    asyncio.run(main())
'''

    @staticmethod
    def read_manifest(package_path: str | Path) -> dict[str, Any]:
        """Read the manifest from a ``.zip`` package without extracting."""
        with zipfile.ZipFile(package_path, "r") as zf:
            return json.loads(zf.read("manifest.json").decode("utf-8"))
