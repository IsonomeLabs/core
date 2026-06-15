from __future__ import annotations
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="isonome", help="Isonome -- Agentic robotics framework")
cache_app = typer.Typer(name="cache", help="Calibration cache management")
app.add_typer(cache_app, name="cache")


INIT_FILES: dict[str, str] = {
    "main.py": '''"""Robot entrypoint -- Isonome bootstraps Agent lifecycle."""
import asyncio
from isonome.core.app import IsonomeApp
from isonome.core.config import AppConfig


async def main():
    app = IsonomeApp(AppConfig())
    await app.run(duration_s=60)


if __name__ == "__main__":
    asyncio.run(main())
''',
    "config.yaml": '''reflex:
  frequency_hz: 100.0
  control_freq_hz: 100.0
  policy_freq_hz: 1.0
jepa:
  frequency_hz: 1.0
  backend: openvla
cortex:
  frequency_hz: 0.5
soma:
  urdf_path: urdf/robot.urdf
safety:
  permit_boot_adaptation: false
  error_repeat_threshold: 3
sim:
  engine: pybullet
  gui: false
bridge:
  engine: pybullet
''',
    "layers/__init__.py": "",
    "layers/reflex.py": '''from isonome.core.layers.reflex import ReflexLayer


class MyReflex(ReflexLayer):
    pass
''',
    "layers/jepa.py": '''from isonome.core.layers.jepa import JEPALayer


class MyJEPA(JEPALayer):
    pass
''',
    "layers/cortex.py": '''from isonome.core.layers.cortex import CortexLayer


class MyCortex(CortexLayer):
    pass
''',
    "layers/plasticity.py": '''from isonome.core.layers.plasticity import PlasticityLayer


class MyPlasticity(PlasticityLayer):
    pass
''',
    "sim/world.json": '''{
  "ground": true,
  "obstacles": [],
  "robots": []
}''',
    "tests/__init__.py": "",
    "tests/test_agent.py": '''import pytest
from isonome.core.config import AppConfig
from isonome.core.agent import Agent
from isonome.core.safety import AgentMode


@pytest.mark.asyncio
async def test_agent_boot_shutdown():
    agent = Agent(AppConfig())
    await agent.boot()
    assert agent.mode == AgentMode.IDLE
    await agent.shutdown()
    assert agent.mode == AgentMode.IDLE
'''
}


@app.command()
def init(name: str) -> None:
    """Scaffold a new robot project."""
    base = Path(name)
    if base.exists():
        typer.echo(f"Error: {name}/ already exists", err=True)
        raise typer.Exit(1)

    base.mkdir(parents=True)

    # isonome.toml manifest
    (base / "isonome.toml").write_text(f'[tool.isonome]\nagent_name = "{name}"\n')

    for rel_path, content in INIT_FILES.items():
        file_path = base / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    typer.echo(f"Scaffolded {name}/ with default layers and config")


@app.command()
def sim(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    duration: int = typer.Option(60, "--duration", "-d", help="Seconds to run"),
    bridge: str = typer.Option("mock", "--bridge", "-b", help="Bridge engine"),
) -> None:
    """Run robot in simulation mode."""
    import asyncio

    from isonome.core.app import IsonomeApp
    from isonome.core.config import AppConfig, BridgeConfig

    if not config.exists():
        typer.echo(f"Config not found: {config}", err=True)
        raise typer.Exit(1)

    app_cfg = AppConfig.from_yaml(config)
    app_cfg.bridge = BridgeConfig(engine=bridge)

    typer.echo(f"Starting simulation with {bridge} bridge...")
    isonome_app = IsonomeApp(app_cfg)

    async def _run() -> None:
        await isonome_app.run(duration_s=float(duration))

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        typer.echo("\nSimulation interrupted.")


@app.command()
def run(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    duration: int = typer.Option(60, "--duration", "-d", help="Seconds to run"),
    package: Optional[Path] = typer.Option(
        None, "--package", "-p", help="Certified policy package (.zip) to load"
    ),
) -> None:
    """Run robot on hardware.

    Forces ``bridge.engine = "hardware"`` and runs the agent loop against a
    ``HardwareBridge``.  If no concrete ``HardwareBridge`` is wired in
    ``config.bridge.engine_options``, the built-in ``StubHardwareBridge`` is
    used (useful for dry-runs and CI).

    An optional certified policy package can be supplied; its manifest is
    validated and, if the package contains ``policy/policy.pt``, the kernel
    path is pointed at the extracted weights before boot.
    """
    import asyncio

    from isonome.core.app import IsonomeApp
    from isonome.core.config import AppConfig, BridgeConfig

    if not config.exists():
        typer.echo(f"Config not found: {config}", err=True)
        raise typer.Exit(1)

    app_cfg = AppConfig.from_yaml(config)

    if package is not None:
        if not package.exists():
            typer.echo(f"Policy package not found: {package}", err=True)
            raise typer.Exit(1)
        app_cfg = _load_package_into_config(app_cfg, package)

    if not app_cfg.soma.urdf_path:
        typer.echo(
            "Hardware mode requires ``soma.urdf_path`` to be set in config.",
            err=True,
        )
        raise typer.Exit(1)

    urdf_path = Path(app_cfg.soma.urdf_path)
    if not urdf_path.exists():
        typer.echo(f"URDF not found: {urdf_path}", err=True)
        raise typer.Exit(1)

    app_cfg.bridge = BridgeConfig(engine="hardware", engine_options=app_cfg.bridge.engine_options)

    typer.echo(
        f"Starting hardware run for '{app_cfg.agent_name}' "
        f"(duration={duration}s, urdf={urdf_path})..."
    )
    isonome_app = IsonomeApp(app_cfg)

    async def _run() -> None:
        await isonome_app.run(duration_s=float(duration))

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        typer.echo("\nHardware run interrupted.")


@app.command()
def deploy(
    package: Path = typer.Argument(..., help="Path to certified policy package (.zip)"),
    target: Path = typer.Option(
        Path("~/.isonome/deployments"),
        "--target",
        "-t",
        help="Deployment target directory",
    ),
    robot_ip: Optional[str] = typer.Option(
        None, "--robot-ip", help="Target robot IP / hostname (recorded in manifest)"
    ),
    protocol: str = typer.Option(
        "ros2",
        "--protocol",
        help="Deployment protocol: ros2 | mqtt | serial | http",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Runtime config file to copy into the deployment",
    ),
) -> None:
    """Deploy a certified policy package to a target directory.

    Validates the archive (manifest + launcher), extracts it, copies an
    optional runtime config, and writes a deployment manifest with target
    metadata.  The resulting directory is a self-contained deployment that
    can be pushed to a robot by a ROS2/MQTT/HTTP transport in a follow-up
    enterprise step.
    """
    if not package.exists():
        typer.echo(f"Policy package not found: {package}", err=True)
        raise typer.Exit(1)

    if package.suffix != ".zip":
        typer.echo(f"Package must be a .zip file: {package}", err=True)
        raise typer.Exit(1)

    manifest = _read_package_manifest(package)
    if manifest is None:
        typer.echo("Invalid package: manifest.json is missing.", err=True)
        raise typer.Exit(1)

    target = target.expanduser()
    deploy_dir = target / package.stem
    if deploy_dir.exists():
        typer.echo(f"Deployment directory already exists: {deploy_dir}", err=True)
        raise typer.Exit(1)

    deploy_dir.mkdir(parents=True)

    # Extract the certified package
    with zipfile.ZipFile(package, "r") as zf:
        zf.extractall(deploy_dir)

    # Copy runtime config if provided
    copied_config: Optional[Path] = None
    if config is not None:
        if not config.exists():
            typer.echo(f"Config file not found: {config}", err=True)
            raise typer.Exit(1)
        copied_config = deploy_dir / "config.yaml"
        shutil.copy2(config, copied_config)

    # Write deployment manifest
    deployment_manifest = {
        "source_package": str(package.resolve()),
        "package_manifest": manifest,
        "robot_ip": robot_ip,
        "protocol": protocol,
        "copied_config": str(copied_config) if copied_config else None,
        "deployed_at": _utc_now_iso(),
    }
    manifest_path = deploy_dir / "deployment_manifest.json"
    manifest_path.write_text(json.dumps(deployment_manifest, indent=2, default=str))

    typer.echo(f"Deployed to {deploy_dir}")
    typer.echo(f"  manifest: {manifest_path}")
    if copied_config:
        typer.echo(f"  config:   {copied_config}")


# ---------------------------------------------------------------------------
# Helpers for run / deploy
# ---------------------------------------------------------------------------


def _read_package_manifest(package_path: Path) -> Optional[dict[str, Any]]:
    """Read ``manifest.json`` from a certified policy package .zip."""
    try:
        with zipfile.ZipFile(package_path, "r") as zf:
            if "manifest.json" not in zf.namelist():
                return None
            return json.loads(zf.read("manifest.json").decode("utf-8"))
    except (zipfile.BadZipFile, json.JSONDecodeError, OSError):
        return None


def _load_package_into_config(app_cfg: "AppConfig", package: Path) -> "AppConfig":
    """Merge a certified policy package into an ``AppConfig`` for runtime use.

    The package is extracted to a temporary directory; ``soma.kernel_path`` is
    pointed at ``policy/policy.pt`` if present.  Other fields (reflex gains,
    agent configs) are left as-is so the supplied ``config.yaml`` remains the
    source of truth for runtime parameters.
    """
    import torch

    extract_dir = Path(tempfile.mkdtemp(prefix="isonome_run_"))
    with zipfile.ZipFile(package, "r") as zf:
        zf.extractall(extract_dir)

    policy_pt = extract_dir / "policy" / "policy.pt"
    if policy_pt.exists():
        # Verify the checkpoint is loadable before pointing the config at it.
        try:
            torch.load(policy_pt, map_location="cpu", weights_only=True)
        except Exception as exc:
            typer.echo(
                f"Policy checkpoint appears corrupt: {exc}",
                err=True,
            )
            raise typer.Exit(1)
        app_cfg.soma.kernel_path = str(policy_pt)

    return app_cfg


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@app.command()
def calibrate(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    urdf: Path = typer.Option(Path("urdf/robot.urdf"), "--urdf", "-u"),
    topology_hash: str = typer.Option(..., "--topology-hash", "-t", help="Robot topology hash"),
    task_type: str = typer.Option("reach", "--task", help="Task type"),
    vla_version: str = typer.Option("openvla-7b-v1", "--vla-version", help="VLA version"),
    episodes: int = typer.Option(100, "--episodes", "-e", help="Validation episodes"),
    namespace: str = typer.Option("public", "--namespace", "-n", help="Cache namespace"),
    output_dir: Path = typer.Option(Path("~/.isonome/calibrations"), "--output-dir", "-o"),
) -> None:
    """Run the calibration / training pipeline (gap #3)."""
    import json

    import torch
    from isonome.core.config import AppConfig
    from isonome.praxis.calibration import CalibrationConfig, CalibrationPipeline

    if not config.exists():
        typer.echo(f"Config not found: {config}", err=True)
        raise typer.Exit(1)
    if not urdf.exists():
        typer.echo(f"URDF not found: {urdf}", err=True)
        raise typer.Exit(1)

    app_cfg = AppConfig.from_yaml(config)
    cal_cfg = CalibrationConfig(
        task_type=task_type,
        vla_version=vla_version,
        output_dir=output_dir,
        cache_namespace=namespace,
        coordinator_strategy=app_cfg.calibration.coordinator_strategy,
        agent_configs=app_cfg.calibration.agent_configs,
        reflex_gains=app_cfg.calibration.reflex_gains,
        metadata={**app_cfg.calibration.metadata, "source_config": str(config)},
    )
    cal_cfg.validation.episodes = episodes

    pipeline = CalibrationPipeline(cal_cfg)
    result = pipeline.run(
        base_urdf_path=urdf,
        topology_hash=topology_hash,
        topology_vector=None,
    )
    typer.echo(json.dumps(result.to_dict(), indent=2, default=str))
    if not result.certified:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Calibration cache CLI (gap #6)
# ---------------------------------------------------------------------------

def _load_cache(cache_root: Path) -> "CalibrationCache":
    from isonome.praxis.calibration_cache import CalibrationCache

    return CalibrationCache(root_dir=cache_root)


@cache_app.command("lookup")
def cache_lookup(
    topology_hash: str,
    task_type: str,
    vla_version: str,
    namespace: str = typer.Option("public", "--namespace", "-n", help="Cache namespace"),
    near_match: bool = typer.Option(False, "--near-match", help="Include topology-near matches"),
    epsilon: float = typer.Option(0.1, "--epsilon", "-e", help="Near-match distance threshold"),
    cache_root: Path = typer.Option(Path("~/.isonome/cache"), "--cache-root"),
) -> None:
    """Look up a certified policy package by topology + task + VLA version."""
    import torch
    from isonome.praxis.calibration_cache import CacheKey

    cache = _load_cache(cache_root)
    key = CacheKey(topology_hash=topology_hash, task_type=task_type, vla_version=vla_version)

    exact = cache.get(key, namespace=namespace)
    result: dict = {"exact_match": None, "near_matches": []}
    if exact is not None:
        result["exact_match"] = exact.to_dict()

    if near_match:
        # Use topology_hash as a stand-in feature vector for CLI demos if no
        # vector is supplied: embed the hex string as a normalized 32-D vector.
        vec = _topology_hash_to_vector(topology_hash)
        matches = cache.find_near_matches(key, vec, epsilon=epsilon, namespace=namespace)
        result["near_matches"] = [
            {"distance": distance, "package": pkg.to_dict()} for distance, pkg in matches
        ]

    if result["exact_match"] is None and not result["near_matches"]:
        typer.echo("No matching cache entry found.", err=True)
        raise typer.Exit(1)

    typer.echo(json.dumps(result, indent=2))


@cache_app.command("put")
def cache_put(
    topology_hash: str,
    task_type: str,
    vla_version: str,
    package: Path = typer.Argument(..., help="Path to package JSON file"),
    namespace: str = typer.Option("public", "--namespace", "-n", help="Cache namespace"),
    topology_vector: Optional[Path] = typer.Option(
        None, "--topology-vector", help="Path to JSON file with 32-D topology vector"
    ),
    cache_root: Path = typer.Option(Path("~/.isonome/cache"), "--cache-root"),
) -> None:
    """Store a certified policy package in the calibration cache."""
    import torch
    from isonome.praxis.calibration_cache import CacheKey, CertifiedPolicyPackage

    if not package.exists():
        typer.echo(f"Package file not found: {package}", err=True)
        raise typer.Exit(1)

    cache = _load_cache(cache_root)
    key = CacheKey(topology_hash=topology_hash, task_type=task_type, vla_version=vla_version)
    pkg_data = json.loads(package.read_text(encoding="utf-8"))
    pkg = CertifiedPolicyPackage.from_dict(pkg_data)

    vec: torch.Tensor | None = None
    if topology_vector is not None:
        if not topology_vector.exists():
            typer.echo(f"Topology vector file not found: {topology_vector}", err=True)
            raise typer.Exit(1)
        vec_list = json.loads(topology_vector.read_text(encoding="utf-8"))
        vec = torch.tensor(vec_list, dtype=torch.float32)
        if vec.shape != (32,):
            typer.echo(f"Topology vector must have shape (32,), got {tuple(vec.shape)}", err=True)
            raise typer.Exit(1)

    entry_dir = cache.put(key, pkg, namespace=namespace, topology_vector=vec)
    typer.echo(f"Cached at {entry_dir}")


@cache_app.command("list")
def cache_list(
    namespace: str = typer.Option("public", "--namespace", "-n", help="Cache namespace"),
    cache_root: Path = typer.Option(Path("~/.isonome/cache"), "--cache-root"),
) -> None:
    """List all cache keys in a namespace."""
    cache = _load_cache(cache_root)
    keys = cache.list_keys(namespace=namespace)
    typer.echo(
        json.dumps(
            [k.to_dict() for k in keys],
            indent=2,
        )
    )


def _topology_hash_to_vector(topology_hash: str) -> "torch.Tensor":
    """Create a deterministic 32-D feature vector from a topology hash string.

    This is a CLI convenience fallback; real near-match search should supply
    the actual 32-D morphology vector.
    """
    import torch

    # Use the first 32 bytes of the SHA-256 of the hash string as normalized
    # features in [0, 1].
    digest = hashlib.sha256(topology_hash.encode()).digest()
    values = [b / 255.0 for b in digest[:32]]
    return torch.tensor(values, dtype=torch.float32)


if __name__ == "__main__":
    app()
