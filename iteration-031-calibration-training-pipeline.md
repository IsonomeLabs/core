# Iteration 031: Calibration / Training Pipeline — Gap #3 Foundation

**Date**: 2026-06-15
**Status**: Complete
**Test Coverage**: 14 new tests (all passing)
**Backward Compatibility**: Full — no existing interfaces changed; new package only.

---

## Problem

Architecture gap #3: the PRD describes a full simulation/calibration pipeline
(URDF stripper, domain randomization, Isaac Lab parallel envs, CMA-ES /
differentiable sim, composition validation, auto-adjustment, and certified
policy package `.zip` export). None of this existed in the open-source runtime.

## Solution

Introduced `isonome/praxis/calibration/` — a backend-agnostic calibration
pipeline that implements every open-source-friendly piece of gap #3.

### `isonome/praxis/calibration/urdf_stripper.py`

- `URDFStripper` parses URDF with `xml.etree.ElementTree`.
- Keeps only requested joints/links and transitively referenced links.
- Optional removal of transmissions, sensors, and unused gazebo tags.
- `list_joints()` introspection helper.

### `isonome/praxis/calibration/domain_randomization.py`

- `DomainRandomizer` modifies URDF in memory.
- Scales link mass, contact friction (URDF + ODE/MuJoCo forms), and joint damping.
- Produces a lighting override dict for future Isaac Sim / renderer integration.
- Deterministic given a seed.

### `isonome/praxis/calibration/optimizer.py`

- `BlackBoxObjective` interface for pluggable objectives.
- `CMAESOptimizer` — a lightweight, dependency-free CMA-ES implementation.
  - Configurable population, generations, initial sigma, fitness target, seed.
  - Eigendecomposition-based sampling for stability across PyTorch versions.
  - Stabilized covariance updates.
- `OptimizationResult` captures best params, fitness, history, and convergence.

### `isonome/praxis/calibration/validator.py`

- `EpisodeResult` per-episode outcome.
- `CompositionValidator` runs N episodes via an injectable `episode_runner`.
- `ValidationReport` aggregates success rate, mean error, and certification status.

### `isonome/praxis/calibration/exporter.py`

- `PolicyPackageExporter` creates a `.zip` Certified Policy Package.
- Archive contains: `manifest.json`, `agent_configs.json`,
  `coordinator_config.json`, `reflex_gains.json`, `sim_metrics.json`,
  `metadata.json`, `policy/policy.pt`, `launcher.py`, and optional
  `certification_video.mp4`.
- `read_manifest()` inspects a package without extracting.

### `isonome/praxis/calibration/pipeline.py`

- `CalibrationPipeline` orchestrates the full loop:
  1. Strip URDF per agent.
  2. Apply domain randomization.
  3. Run CMA-ES policy optimization.
  4. Validate composition.
  5. Auto-adjust domain randomization strength and validation budget if needed
     (max 5 iterations).
  6. Export `.zip` Certified Policy Package.
  7. Store the package in `CalibrationCache` keyed by topology + task + VLA.
- Includes a `MockPendulumObjective` / `MockEpisodeRunner` default so the
  pipeline is runnable out of the box without Isaac Lab.
- `CalibrationResult` summarizes certification, paths, cache key, and metrics.

### `isonome/praxis/calibration/config.py`

- Dataclass configs: `URDFStripperConfig`, `DomainRandomizationConfig`,
  `CMAESConfig`, `ValidationConfig`, `AutoAdjustmentConfig`, and top-level
  `CalibrationConfig`.

### `isonome/core/config.py`

- Added `CalibrationPipelineConfig` Pydantic model to `AppConfig`.

### `isonome/cli.py`

- Added `isonome calibrate` command with options for config, URDF, topology
  hash, task type, VLA version, validation episodes, cache namespace, and
  output directory.

### `architecture-gaps.md`

- Updated gap #3 from "❌ Missing entirely" to "⚠️ PARTIALLY CLOSED".
- Summary table now shows CMA-ES/auto-adjustment and `.zip` export as
  implemented.

## Test Coverage

14 tests in `tests/test_calibration_pipeline.py`:

| Test | What it covers |
|---|---|
| `test_lists_joints` | URDF joint introspection |
| `test_noop_when_no_filters` | URDF stripper passes through unchanged |
| `test_keeps_only_requested_joints_and_links` | Joint/link filtering |
| `test_strip_to_file` | File output |
| `test_randomize_changes_mass` | Domain randomization behavior |
| `test_randomize_to_file` | Randomized URDF output |
| `test_finds_quadratic_optimum` | CMA-ES converges on a quadratic objective |
| `test_converges_when_target_reached` | Fitness-target convergence |
| `test_all_successful_episodes_certify` | Validator certification |
| `test_failing_threshold_not_certified` | Validator threshold enforcement |
| `test_export_and_read_manifest` | `.zip` export + manifest read |
| `test_runs_mock_pipeline` | End-to-end pipeline with caching |
| `test_missing_urdf_raises` | Pipeline error handling |
| `test_calibrate_command_exists` | CLI command registration |

Full suite: 2063 passed (14 new + 2049 existing).

## Files Changed

- `isonome/praxis/calibration/__init__.py` — **NEW**
- `isonome/praxis/calibration/config.py` — **NEW**
- `isonome/praxis/calibration/urdf_stripper.py` — **NEW**
- `isonome/praxis/calibration/domain_randomization.py` — **NEW**
- `isonome/praxis/calibration/optimizer.py` — **NEW**
- `isonome/praxis/calibration/validator.py` — **NEW**
- `isonome/praxis/calibration/exporter.py` — **NEW**
- `isonome/praxis/calibration/pipeline.py` — **NEW**
- `isonome/core/config.py` — Add `CalibrationPipelineConfig`
- `isonome/cli.py` — Add `calibrate` command
- `tests/test_calibration_pipeline.py` — **NEW**: 14 tests
- `architecture-gaps.md` — Mark gap #3 partially closed
- `iteration-031-calibration-training-pipeline.md` — **NEW**: this document

## Next Steps

- **Gap #4** (Simulation Backends Mismatch): Add Isaac Lab / MuJoCo MJX
  backend adapters that the calibration pipeline can use instead of the mock
  objective.
- **Gap #10** (`run` / `deploy` CLI): Implement hardware bridge boot and deploy.
