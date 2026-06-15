"""Tests for the Isonome CLI (architecture gap #10).

Covers the ``run`` and ``deploy`` commands that close the remaining CLI stubs.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from isonome.cli import app


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def example_urdf() -> Path:
    return Path(__file__).parent.parent / "examples" / "robot_arm.urdf"


def _make_config(tmp_path: Path, urdf_path: Path, extra: dict[str, Any] | None = None) -> Path:
    """Write a minimal config.yaml suitable for the run command."""
    cfg: dict[str, Any] = {
        "agent_name": "test_agent",
        "reflex": {
            "frequency_hz": 100.0,
            "control_freq_hz": 100.0,
            "policy_freq_hz": 1.0,
        },
        "jepa": {"frequency_hz": 10.0, "backend": "openvla"},
        "cortex": {"frequency_hz": 0.5},
        "soma": {"urdf_path": str(urdf_path)},
        "bridge": {"engine": "mock"},
    }
    if extra:
        cfg.update(extra)
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(cfg))
    return config_path


def _make_package(
    tmp_path: Path,
    manifest: dict[str, Any] | None = None,
    include_policy: bool = False,
) -> Path:
    """Create a certified policy package .zip for deploy tests."""
    pkg_path = tmp_path / "policy.zip"
    with zipfile.ZipFile(pkg_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest or {"task": "reach"}, indent=2))
        zf.writestr("launcher.py", "# launcher stub\n")
        if include_policy:
            import io
            import torch

            buf = io.BytesIO()
            torch.save({"policy_params": torch.zeros(4)}, buf)
            zf.writestr("policy/policy.pt", buf.getvalue())
    return pkg_path


# ===========================================================================
# run command
# ===========================================================================


def test_run_missing_config(cli_runner: CliRunner) -> None:
    result = cli_runner.invoke(app, ["run", "--config", "missing.yaml"])
    assert result.exit_code == 1
    assert "Config not found" in result.output


def test_run_missing_urdf(cli_runner: CliRunner, tmp_path: Path) -> None:
    config = _make_config(tmp_path, tmp_path / "missing.urdf")
    result = cli_runner.invoke(app, ["run", "--config", str(config)])
    assert result.exit_code == 1
    assert "URDF not found" in result.output


def test_run_hardware_mode_boots_and_ticks(
    cli_runner: CliRunner, tmp_path: Path, example_urdf: Path
) -> None:
    """Run should boot with the stub hardware bridge and complete a short loop."""
    config = _make_config(tmp_path, example_urdf)
    result = cli_runner.invoke(
        app,
        ["run", "--config", str(config), "--duration", "1"],
    )
    assert result.exit_code == 0, result.output
    assert "Starting hardware run" in result.output


def test_run_with_policy_package(
    cli_runner: CliRunner, tmp_path: Path, example_urdf: Path
) -> None:
    """Run should accept a certified policy package and point the kernel at it."""
    config = _make_config(tmp_path, example_urdf)
    pkg = _make_package(tmp_path, include_policy=True)
    result = cli_runner.invoke(
        app,
        [
            "run",
            "--config",
            str(config),
            "--duration",
            "1",
            "--package",
            str(pkg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Starting hardware run" in result.output


def test_run_with_corrupt_policy_package(
    cli_runner: CliRunner, tmp_path: Path, example_urdf: Path
) -> None:
    """Run should reject a package whose policy checkpoint cannot be loaded."""
    config = _make_config(tmp_path, example_urdf)
    pkg = tmp_path / "bad_policy.zip"
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps({"task": "reach"}))
        zf.writestr("policy/policy.pt", b"not a torch checkpoint")
    result = cli_runner.invoke(
        app,
        ["run", "--config", str(config), "--package", str(pkg)],
    )
    assert result.exit_code == 1, result.output
    assert "Policy checkpoint appears corrupt" in result.output


# ===========================================================================
# deploy command
# ===========================================================================


def test_deploy_missing_package(cli_runner: CliRunner, tmp_path: Path) -> None:
    result = cli_runner.invoke(
        app,
        [
            "deploy",
            str(tmp_path / "missing.zip"),
            "--target",
            str(tmp_path / "deployments"),
        ],
    )
    assert result.exit_code == 1
    assert "Policy package not found" in result.output


def test_deploy_invalid_extension(cli_runner: CliRunner, tmp_path: Path) -> None:
    bad_pkg = tmp_path / "policy.tar"
    bad_pkg.write_text("not a zip")
    result = cli_runner.invoke(
        app,
        ["deploy", str(bad_pkg), "--target", str(tmp_path / "deployments")],
    )
    assert result.exit_code == 1
    assert "Package must be a .zip file" in result.output


def test_deploy_missing_manifest(cli_runner: CliRunner, tmp_path: Path) -> None:
    pkg = tmp_path / "empty.zip"
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("readme.txt", "hello")
    result = cli_runner.invoke(
        app,
        ["deploy", str(pkg), "--target", str(tmp_path / "deployments")],
    )
    assert result.exit_code == 1
    assert "manifest.json is missing" in result.output


def test_deploy_success(cli_runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "deployments"
    pkg = _make_package(tmp_path, manifest={"task": "reach red cube"})
    result = cli_runner.invoke(
        app,
        [
            "deploy",
            str(pkg),
            "--target",
            str(target),
            "--robot-ip",
            "192.168.1.100",
            "--protocol",
            "ros2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Deployed to" in result.output

    deploy_dir = target / "policy"
    assert deploy_dir.exists()
    assert (deploy_dir / "manifest.json").exists()
    assert (deploy_dir / "launcher.py").exists()

    deployment_manifest_path = deploy_dir / "deployment_manifest.json"
    assert deployment_manifest_path.exists()
    deployment_manifest = json.loads(deployment_manifest_path.read_text())
    assert deployment_manifest["robot_ip"] == "192.168.1.100"
    assert deployment_manifest["protocol"] == "ros2"
    assert deployment_manifest["package_manifest"]["task"] == "reach red cube"
    assert "deployed_at" in deployment_manifest


def test_deploy_with_config_copy(cli_runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "deployments"
    pkg = _make_package(tmp_path)
    config = tmp_path / "runtime.yaml"
    config.write_text("agent_name: deployed_agent\n")

    result = cli_runner.invoke(
        app,
        [
            "deploy",
            str(pkg),
            "--target",
            str(target),
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 0, result.output

    deploy_dir = target / "policy"
    copied = deploy_dir / "config.yaml"
    assert copied.exists()
    assert "deployed_agent" in copied.read_text()

    deployment_manifest = json.loads((deploy_dir / "deployment_manifest.json").read_text())
    assert deployment_manifest["copied_config"] == str(copied)


def test_deploy_target_already_exists(cli_runner: CliRunner, tmp_path: Path) -> None:
    target = tmp_path / "deployments"
    pkg = _make_package(tmp_path)
    (target / "policy").mkdir(parents=True)
    result = cli_runner.invoke(
        app,
        ["deploy", str(pkg), "--target", str(target)],
    )
    assert result.exit_code == 1
    assert "Deployment directory already exists" in result.output
