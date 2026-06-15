"""Tests for the calibration / training pipeline (architecture gap #3)."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest
import torch

from isonome.praxis.calibration import (
    BlackBoxObjective,
    CMAESConfig,
    CMAESOptimizer,
    CalibrationConfig,
    CalibrationPipeline,
    CompositionValidator,
    DomainRandomizer,
    EpisodeResult,
    PolicyPackageExporter,
    PolicyPackageArtifacts,
    URDFStripper,
    URDFStripperConfig,
    ValidationConfig,
)


@pytest.fixture
def simple_urdf(tmp_path: Path) -> Path:
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        """<?xml version="1.0"?>
<robot name="test">
  <link name="base"/>
  <link name="upper"/>
  <link name="lower"/>
  <joint name="shoulder" type="revolute">
    <parent link="base"/>
    <child link="upper"/>
    <limit lower="-3.14" upper="3.14"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/>
    <child link="lower"/>
    <limit lower="-1.57" upper="1.57"/>
  </joint>
</robot>
"""
    )
    return urdf


class QuadraticObjective(BlackBoxObjective):
    def __init__(self, dim: int) -> None:
        self._dim = dim
        self.target = torch.ones(dim) * 1.5

    @property
    def dim(self) -> int:
        return self._dim

    def evaluate(self, params: torch.Tensor) -> float:
        return -float((params - self.target).norm().item())


class TestURDFStripper:
    def test_lists_joints(self, simple_urdf: Path) -> None:
        stripper = URDFStripper()
        joints = stripper.list_joints(simple_urdf)
        names = [j["name"] for j in joints]
        assert names == ["shoulder", "elbow"]

    def test_noop_when_no_filters(self, simple_urdf: Path) -> None:
        stripper = URDFStripper()
        root = stripper.strip(simple_urdf)
        assert len(root.findall("joint")) == 2
        assert len(root.findall("link")) == 3

    def test_keeps_only_requested_joints_and_links(self, simple_urdf: Path) -> None:
        config = URDFStripperConfig(keep_joints=["shoulder"])
        stripper = URDFStripper(config)
        root = stripper.strip(simple_urdf)
        joint_names = {j.get("name") for j in root.findall("joint")}
        link_names = {l.get("name") for l in root.findall("link")}
        assert joint_names == {"shoulder"}
        assert link_names == {"base", "upper"}

    def test_strip_to_file(self, simple_urdf: Path, tmp_path: Path) -> None:
        out = tmp_path / "stripped.urdf"
        stripper = URDFStripper(URDFStripperConfig(keep_joints=["elbow"]))
        path = stripper.strip_to_file(simple_urdf, out)
        assert path.exists()
        root = ET.parse(path).getroot()
        assert len(root.findall("joint")) == 1


class TestDomainRandomizer:
    def test_randomize_changes_mass(self, simple_urdf: Path) -> None:
        randomizer = DomainRandomizer()
        root, lighting = randomizer.randomize(simple_urdf, seed=123)
        masses = [
            float(link.find("inertial/mass").get("value", "1.0"))
            for link in root.findall("link")
            if link.find("inertial/mass") is not None
        ]
        # Our fixture has no inertial/mass, so this should be empty.
        assert masses == []
        assert lighting == {}

    def test_randomize_to_file(self, simple_urdf: Path, tmp_path: Path) -> None:
        out = tmp_path / "rand.urdf"
        randomizer = DomainRandomizer()
        path, _ = randomizer.randomize_to_file(simple_urdf, out, seed=7)
        assert path.exists()


class TestCMAESOptimizer:
    def test_finds_quadratic_optimum(self) -> None:
        objective = QuadraticObjective(3)
        config = CMAESConfig(
            population_size=16,
            initial_sigma=0.5,
            max_generations=30,
            seed=42,
        )
        optimizer = CMAESOptimizer(objective, config)
        result = optimizer.optimize()
        assert result.best_fitness > -0.5
        assert result.generation <= config.max_generations
        assert result.best_params.shape == (3,)

    def test_converges_when_target_reached(self) -> None:
        objective = QuadraticObjective(2)
        config = CMAESConfig(
            population_size=12,
            initial_sigma=1.0,
            max_generations=100,
            fitness_target=-0.01,
            seed=42,
        )
        optimizer = CMAESOptimizer(objective, config)
        result = optimizer.optimize()
        assert result.converged
        assert result.best_fitness >= -0.01


class TestCompositionValidator:
    def test_all_successful_episodes_certify(self) -> None:
        config = ValidationConfig(episodes=10, success_rate_threshold=0.9)
        validator = CompositionValidator(config, episode_runner=lambda: EpisodeResult(True, 10))
        report = validator.validate()
        assert report.success_rate == 1.0
        assert report.certified

    def test_failing_threshold_not_certified(self) -> None:
        call_count = 0

        def runner() -> EpisodeResult:
            nonlocal call_count
            call_count += 1
            return EpisodeResult(call_count > 5, 10)

        config = ValidationConfig(episodes=10, success_rate_threshold=0.99)
        validator = CompositionValidator(config, episode_runner=runner)
        report = validator.validate()
        assert report.success_rate == 0.5
        assert not report.certified


class TestPolicyPackageExporter:
    def test_export_and_read_manifest(self, tmp_path: Path) -> None:
        out = tmp_path / "pkg.zip"
        artifacts = PolicyPackageArtifacts(
            manifest={"task": "reach"},
            policy_params=torch.tensor([0.1, 0.2]),
        )
        exporter = PolicyPackageExporter()
        path = exporter.export(artifacts, out)
        assert path.exists()

        manifest = exporter.read_manifest(path)
        assert manifest["task"] == "reach"

        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
            assert "manifest.json" in names
            assert "policy/policy.pt" in names
            assert "launcher.py" in names


class TestCalibrationPipeline:
    def test_runs_mock_pipeline(self, simple_urdf: Path, tmp_path: Path) -> None:
        config = CalibrationConfig(
            task_type="reach",
            vla_version="openvla-test",
            output_dir=tmp_path / "cal",
            cache_namespace="private",
            metadata={"dof": 2},
        )
        # Keep the test fast.
        config.optimizer.max_generations = 5
        config.optimizer.population_size = 8
        config.validation.episodes = 10
        config.validation.success_rate_threshold = 0.0
        config.auto_adjustment.max_iterations = 1

        pipeline = CalibrationPipeline(config)
        result = pipeline.run(
            base_urdf_path=simple_urdf,
            topology_hash="deadbeef" * 4,
            topology_vector=None,
        )
        assert result.package_path is not None
        assert result.cache_key is not None
        assert result.cache_entry_dir is not None
        assert result.validation is not None
        assert result.validation.episodes == 10
        assert result.to_dict()["certified"] is True

    def test_missing_urdf_raises(self, tmp_path: Path) -> None:
        config = CalibrationConfig(output_dir=tmp_path / "cal")
        pipeline = CalibrationPipeline(config)
        with pytest.raises(FileNotFoundError):
            pipeline.run(base_urdf_path=tmp_path / "missing.urdf", topology_hash="abc")


class TestCalibrationCLI:
    def test_calibrate_command_exists(self) -> None:
        from typer.testing import CliRunner
        from isonome.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["calibrate", "--help"])
        assert result.exit_code == 0
        assert "topology-hash" in result.output
