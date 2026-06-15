"""URDF Stripper — extract per-agent joint subsets from a base URDF.

Architecture gap #3, Step 1: ``URDF Stripper`` produces isolated URDFs that
contain only the joints/links required by a single agent.  This keeps parallel
calibration environments small and prevents one agent's morphology from leaking
into another's policy.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from isonome.praxis.calibration.config import URDFStripperConfig
from isonome.utils.logging import get_layer_logger


class URDFStripper:
    """Strip a URDF down to a requested subset of joints and links.

    Parameters
    ----------
    config:
        Stripping rules.  If both ``keep_joints`` and ``keep_links`` are
        ``None``, the URDF is returned unchanged (useful for monolithic
        single-agent robots).
    """

    def __init__(self, config: URDFStripperConfig | None = None) -> None:
        self._config = config or URDFStripperConfig()
        self._logger = get_layer_logger("praxis.calibration.urdf_stripper")

    def strip(
        self,
        urdf_path: str | Path,
        *,
        keep_joints: list[str] | None = None,
        keep_links: list[str] | None = None,
    ) -> ET.Element:
        """Return a stripped URDF root element.

        Parameters
        ----------
        urdf_path:
            Path to the source URDF.
        keep_joints:
            Override for which joint names to retain.  Defaults to
            ``config.keep_joints``.
        keep_links:
            Override for which link names to retain.  If omitted, the links
            referenced by ``keep_joints`` are automatically retained.

        Returns
        -------
        A deep-copied ``ElementTree`` root that can be serialized with
        ``ElementTree.write()``.
        """
        keep_joints = keep_joints or self._config.keep_joints
        keep_links = keep_links or self._config.keep_links

        tree = ET.parse(Path(urdf_path))
        root = copy.deepcopy(tree.getroot())

        if keep_joints is None and keep_links is None:
            self._logger.info("urdf_strip_noop", extra={"urdf": str(urdf_path)})
            return root

        # Build the transitive closure of links we must keep.
        keep_link_set: set[str] = set()
        if keep_links:
            keep_link_set.update(keep_links)

        joints_to_keep: list[ET.Element] = []
        for joint in root.findall("joint"):
            name = joint.get("name", "")
            if keep_joints is None or name in keep_joints:
                joints_to_keep.append(joint)
                parent = joint.find("parent")
                child = joint.find("child")
                if parent is not None:
                    keep_link_set.add(parent.get("link", ""))
                if child is not None:
                    keep_link_set.add(child.get("link", ""))

        # Remove joints we are not keeping.
        for joint in list(root.findall("joint")):
            if joint not in joints_to_keep:
                root.remove(joint)

        # Remove links we are not keeping.
        for link in list(root.findall("link")):
            name = link.get("name", "")
            if name not in keep_link_set:
                root.remove(link)

        # Optional: strip transmissions and sensors.
        if self._config.remove_transmissions:
            for transmission in list(root.findall("transmission")):
                root.remove(transmission)
        if self._config.remove_sensors:
            for sensor in list(root.findall("sensor")):
                root.remove(sensor)
            for gazebo in list(root.findall("gazebo")):
                if gazebo.get("reference") not in keep_link_set:
                    root.remove(gazebo)

        self._logger.info(
            "urdf_strip",
            extra={
                "urdf": str(urdf_path),
                "kept_joints": len(joints_to_keep),
                "kept_links": len(keep_link_set),
            },
        )
        return root

    def strip_to_file(
        self,
        urdf_path: str | Path,
        output_path: str | Path,
        *,
        keep_joints: list[str] | None = None,
        keep_links: list[str] | None = None,
    ) -> Path:
        """Strip a URDF and write the result to ``output_path``."""
        root = self.strip(urdf_path, keep_joints=keep_joints, keep_links=keep_links)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
        return output_path

    def list_joints(self, urdf_path: str | Path) -> list[dict[str, Any]]:
        """Return a list of non-fixed joints in the URDF."""
        tree = ET.parse(Path(urdf_path))
        root = tree.getroot()
        joints: list[dict[str, Any]] = []
        for joint in root.findall("joint"):
            jtype = joint.get("type", "fixed")
            if jtype == "fixed":
                continue
            limit = joint.find("limit")
            joints.append(
                {
                    "name": joint.get("name", ""),
                    "type": jtype,
                    "parent": joint.find("parent").get("link", "") if joint.find("parent") is not None else "",
                    "child": joint.find("child").get("link", "") if joint.find("child") is not None else "",
                    "lower": float(limit.get("lower", "-3.14159")) if limit is not None else None,
                    "upper": float(limit.get("upper", "3.14159")) if limit is not None else None,
                }
            )
        return joints
