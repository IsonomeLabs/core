# Iteration 033: CLI `run` and `deploy` Commands — Gap #10

**Date**: 2026-06-15
**Status**: Complete
**Test Coverage**: 11 new tests (all passing)
**Backward Compatibility**: Full — no existing interfaces changed; only CLI stubs replaced.

---

## Problem

Architecture gap #10: the PRD / architecture diagram specifies a Typer CLI with
`init | sim | run | deploy`.  `init`, `sim`, and `cache` were already
implemented, but `run` and `deploy` were still stubs printing placeholder text.

## Solution

Replaced the stubs in `isonome/cli.py` with full implementations and added
comprehensive tests in `tests/test_cli.py`.

### `isonome run`

- Loads `config.yaml` (or `--config`).
- Validates that `soma.urdf_path` is set and the URDF file exists.
- Forces `bridge.engine = "hardware"` so the agent uses the hardware body-bridge
  path regardless of the config's bridge setting.
- Boots `IsonomeApp` and runs the agent loop for `--duration` seconds.
- Optional `--package` accepts a certified policy package `.zip` produced by
  `isonome calibrate`.  The package is extracted to a temp directory and, if it
  contains `policy/policy.pt`, `soma.kernel_path` is pointed at it after a
  loadability check.
- If no concrete `HardwareBridge` is wired, the existing
  `HardwareBridgeAdapter` falls back to `StubHardwareBridge`, making the command
  testable and usable as a dry-run.

### `isonome deploy`

- Accepts a certified policy package `.zip` as an argument.
- Validates that the file exists, is a `.zip`, and contains `manifest.json`.
- Extracts the archive into a deployment directory (`--target/{package_stem}`).
- Optionally copies a runtime config (`--config`) into the deployment.
- Writes a `deployment_manifest.json` recording:
  - source package path,
  - package manifest,
  - target `robot_ip` (`--robot-ip`),
  - transport protocol (`--protocol`, default `ros2`),
  - copied config path,
  - deployment timestamp.
- Fails early if the target deployment directory already exists.

The resulting directory is a self-contained deployment artifact.  The actual
network push to a physical robot (ROS2/MQTT/serial/HTTP) is intentionally left
as a follow-up enterprise step; this closes the open-source runtime portion of
gap #10.

### `tests/test_cli.py`

New test suite covering:

- `run` with missing config.
- `run` with missing URDF.
- `run` boots and ticks with the stub hardware bridge.
- `run` accepts a valid certified policy package.
- `run` rejects a corrupt policy checkpoint.
- `deploy` with missing package.
- `deploy` with non-`.zip` file.
- `deploy` with package missing `manifest.json`.
- `deploy` successful extraction and manifest generation.
- `deploy` with runtime config copy.
- `deploy` target-directory collision.

## Files Changed

- `isonome/cli.py` — replaced `run()` and `deploy()` stubs; added helper
  functions `_read_package_manifest`, `_load_package_into_config`, and
  `_utc_now_iso`.
- `tests/test_cli.py` — new test module (11 tests).
- `architecture-gaps.md` — marked gap #10 as closed and updated summary table.

## Verification

```bash
.venv/bin/python -m pytest tests/test_cli.py -v
```

All 11 tests pass.
