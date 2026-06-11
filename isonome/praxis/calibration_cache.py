"""Calibration Cache — topology-aware policy package cache.

Addresses architecture gap #6: the architecture specifies a ``Calibration
Cache`` keyed by ``SHA256(topology_hash + task_type + vla_version)`` that
stores certified policy packages.  Previously only a generic ``SemanticCache``
(for Cortex advice strings) existed in ``isonome/llm/cache.py``.

This module implements the cache with:

* Exact-match lookup by topology/task/VLA version.
* Near-match search by topology vector distance (L2).
* Public / private namespaces on disk.
* A CLI in ``isonome.cli`` for lookup and management.

Example
-------
>>> from isonome.praxis.calibration_cache import CalibrationCache, CacheKey, CertifiedPolicyPackage
>>> cache = CalibrationCache()
>>> key = CacheKey(topology_hash="abc123", task_type="reach", vla_version="openvla-7b")
>>> pkg = CertifiedPolicyPackage(manifest={"task": "reach"})
>>> cache.put(key, pkg)
>>> cache.get(key)
CertifiedPolicyPackage(manifest={'task': 'reach'}, ...)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from isonome.utils.logging import get_layer_logger


@dataclass(frozen=True)
class CacheKey:
    """Cache key for a certified policy package.

    Composed of ``topology_hash``, ``task_type``, and ``vla_version``,
    matching the architecture spec (Diagram 1) and PRD FR-5.1.
    """

    topology_hash: str
    task_type: str
    vla_version: str

    def sha256(self) -> str:
        """Return the canonical SHA-256 hex digest for this key."""
        payload = f"{self.topology_hash}:{self.task_type}:{self.vla_version}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_hash": self.topology_hash,
            "task_type": self.task_type,
            "vla_version": self.vla_version,
            "sha256": self.sha256(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheKey":
        return cls(
            topology_hash=data["topology_hash"],
            task_type=data["task_type"],
            vla_version=data["vla_version"],
        )


@dataclass
class CertifiedPolicyPackage:
    """A certified policy package as stored in the calibration cache.

    Mirrors the artifact list from PRD FR-4.9 / FR-5.3.  Fields are kept
    intentionally loose (dicts + optional paths) so the cache can store
    packages produced by future calibration pipelines without prescribing
    their exact schema.
    """

    manifest: dict[str, Any] = field(default_factory=dict)
    agent_configs: dict[str, Any] = field(default_factory=dict)
    coordinator_config: dict[str, Any] = field(default_factory=dict)
    reflex_gains: dict[str, Any] = field(default_factory=dict)
    sim_metrics: dict[str, Any] = field(default_factory=dict)
    policy_package_path: Optional[str] = None
    certification_video_path: Optional[str] = None
    launcher_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CertifiedPolicyPackage":
        return cls(**data)


class CalibrationCache:
    """On-disk calibration cache keyed by topology + task + VLA version.

    Parameters
    ----------
    root_dir:
        Directory that will hold per-namespace cache entries.
    default_namespace:
        Namespace used when none is supplied.  ``"public"`` and ``"private"``
        are the canonical namespaces from PRD FR-5.4.  The open-source
        runtime stores both as plain directories; enterprise deployments are
        expected to layer encryption on top of the ``"private"`` namespace.
    """

    DEFAULT_NAMESPACE = "public"
    PRIVATE_NAMESPACE = "private"
    META_FILE = "meta.json"
    PACKAGE_FILE = "package.json"

    def __init__(
        self,
        root_dir: str | Path = "~/.isonome/cache",
        default_namespace: str = DEFAULT_NAMESPACE,
    ) -> None:
        self.root = Path(root_dir).expanduser()
        self.default_namespace = default_namespace
        self._logger = get_layer_logger("praxis.calibration_cache")
        self._ensure_namespace(default_namespace)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(
        self,
        key: CacheKey,
        package: CertifiedPolicyPackage,
        *,
        namespace: Optional[str] = None,
        topology_vector: Optional[torch.Tensor] = None,
    ) -> Path:
        """Store a certified policy package in the cache.

        Returns the path to the on-disk entry directory.
        """
        namespace = namespace or self.default_namespace
        self._ensure_namespace(namespace)
        entry_dir = self._entry_dir(key, namespace)
        entry_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "key": key.to_dict(),
            "created_at": time.time(),
            "namespace": namespace,
            "topology_vector": (
                self._tensor_to_list(topology_vector) if topology_vector is not None else None
            ),
        }

        (entry_dir / self.META_FILE).write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )
        (entry_dir / self.PACKAGE_FILE).write_text(
            json.dumps(package.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

        self._logger.info(
            "cache_put",
            extra={
                "namespace": namespace,
                "key": key.sha256()[:16],
                "task_type": key.task_type,
                "vla_version": key.vla_version,
            },
        )
        return entry_dir

    def get(
        self,
        key: CacheKey,
        *,
        namespace: Optional[str] = None,
    ) -> Optional[CertifiedPolicyPackage]:
        """Exact-match retrieval of a policy package."""
        namespace = namespace or self.default_namespace
        package_path = self._entry_dir(key, namespace) / self.PACKAGE_FILE
        if not package_path.exists():
            return None
        return CertifiedPolicyPackage.from_dict(json.loads(package_path.read_text(encoding="utf-8")))

    def exists(self, key: CacheKey, *, namespace: Optional[str] = None) -> bool:
        """Check whether an exact-match entry exists."""
        namespace = namespace or self.default_namespace
        return (self._entry_dir(key, namespace) / self.PACKAGE_FILE).exists()

    def list_keys(self, namespace: Optional[str] = None) -> list[CacheKey]:
        """Return all cache keys stored in a namespace."""
        namespace = namespace or self.default_namespace
        keys: list[CacheKey] = []
        ns_dir = self.root / namespace
        if not ns_dir.exists():
            return keys
        for entry_dir in ns_dir.iterdir():
            meta_path = entry_dir / self.META_FILE
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                keys.append(CacheKey.from_dict(meta["key"]))
            except (json.JSONDecodeError, KeyError):
                self._logger.warning("cache_corrupt_entry", extra={"entry": str(entry_dir)})
        return keys

    def find_near_matches(
        self,
        key: CacheKey,
        topology_vector: torch.Tensor,
        *,
        epsilon: float = 0.1,
        namespace: Optional[str] = None,
    ) -> list[tuple[float, CertifiedPolicyPackage]]:
        """Near-match search by topology vector L2 distance.

        Returns a list of ``(distance, package)`` tuples sorted by distance,
        including only entries where ``distance <= epsilon``.

        Parameters
        ----------
        key:
            Supplies ``task_type`` and ``vla_version`` filters.  Only entries
            with matching task type and VLA version are considered.
        topology_vector:
            32-D topology feature vector (from ``MorphologyAnalyzer``).
        epsilon:
            Maximum L2 distance for a match.
        namespace:
            Cache namespace to search.
        """
        namespace = namespace or self.default_namespace
        query = torch.as_tensor(topology_vector, dtype=torch.float32).flatten()
        matches: list[tuple[float, CertifiedPolicyPackage]] = []
        ns_dir = self.root / namespace
        if not ns_dir.exists():
            return matches

        for entry_dir in ns_dir.iterdir():
            meta_path = entry_dir / self.META_FILE
            pkg_path = entry_dir / self.PACKAGE_FILE
            if not meta_path.exists() or not pkg_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                # Only match same task_type / vla_version for near-match
                meta_key = meta["key"]
                if (
                    meta_key["task_type"] != key.task_type
                    or meta_key["vla_version"] != key.vla_version
                ):
                    continue

                tv_list = meta.get("topology_vector")
                if tv_list is None:
                    continue
                entry_vec = torch.tensor(tv_list, dtype=torch.float32)
                if entry_vec.shape != query.shape:
                    continue
                distance = float(torch.norm(query - entry_vec))
                if distance <= epsilon:
                    pkg = CertifiedPolicyPackage.from_dict(
                        json.loads(pkg_path.read_text(encoding="utf-8"))
                    )
                    matches.append((distance, pkg))
            except (json.JSONDecodeError, KeyError):
                self._logger.warning("cache_corrupt_entry", extra={"entry": str(entry_dir)})

        matches.sort(key=lambda x: x[0])
        return matches

    def clear(self, namespace: Optional[str] = None) -> int:
        """Remove all entries from a namespace.  Returns number removed."""
        namespace = namespace or self.default_namespace
        ns_dir = self.root / namespace
        if not ns_dir.exists():
            return 0
        removed = 0
        for entry_dir in ns_dir.iterdir():
            if entry_dir.is_dir():
                self._rm_tree(entry_dir)
                removed += 1
        return removed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_namespace(self, namespace: str) -> None:
        (self.root / namespace).mkdir(parents=True, exist_ok=True)

    def _entry_dir(self, key: CacheKey, namespace: str) -> Path:
        return self.root / namespace / key.sha256()

    @staticmethod
    def _tensor_to_list(tensor: torch.Tensor) -> list[float]:
        return torch.as_tensor(tensor).flatten().tolist()

    @staticmethod
    def _rm_tree(path: Path) -> None:
        for child in path.iterdir():
            if child.is_dir():
                CalibrationCache._rm_tree(child)
            else:
                child.unlink()
        path.rmdir()
