# Isonome

> Open-source runtime for embodied AI — Frozen Brain + Learned Nervous System.

Isonome is an open-source framework for running visuomotor policies (VLAs) on real robots. The core thesis: the VLA policy (π0.7, SmolVLA, OpenVLA, etc.) is a **frozen generalist brain**. The framework provides the **nervous system** that adapts any brain to any body through a learned, local kernel.

## Architecture

### The Four Layers

| Layer | Frequency | Role |
|-------|-----------|------|
| **Soma** | ~100Hz | Body interface: RAW perception, kernel correction, actuation |
| **JEPA** | ~1Hz | Frozen VLA policy loader: deliberates canonical action from RAW state |
| **Cortex** | ~0.5Hz | Discrepancy watcher: generates natural-language advice for JEPA |
| **Reflex** | ~100Hz | Interpolates, enforces safety, executes motor commands |

### Control Flow

```
Soma.perceive() → RAW SensorState
                    ↓
Cortex.advise() → CortexAdvice (text)
                    ↓
JEPA.deliberate() → CanonicalActionChunk (frozen policy output)
                    ↓
SomaKernel.apply() → CorrectedMotorCommand (body-specific)
                    ↓
Reflex.process() → SafeMotorCommand[] (interpolated + clamped)
                    ↓
Soma.act() → Robot
```

### Critical Invariants

1. **JEPA never sees corrected or post-execution state.** `SomaLayer.perceive()` returns **RAW** proprioception and camera frames. Never feed post-kernel corrected states into JEPA or the system will oscillate.
2. **The VLA is frozen.** All loaded parameters have `requires_grad=False`. All inference runs under `torch.no_grad()`.
3. **Cortex never touches motors.** It only produces text advice for JEPA.

## Install

```bash
pip install isonome
```

With simulation support:

```bash
pip install isonome[sim]
```

With Physical Intelligence backend (π0, π0-fast):

```bash
pip install isonome[pi0]
```

With generic VLA backends (SmolVLA, OpenVLA):

```bash
pip install isonome[vla]
```

## Quick Start

### Naive Mapping Demo

```bash
python examples/hello_sim.py
```

Shows the agent running with naive mapping only — no calibrated kernel. The robot will show systematic drift (intentional).

### Calibrated Kernel Demo

```bash
python examples/hello_kernel.py
```

Loads a pre-saved dummy kernel and shows the systematic bias being corrected. Side-by-side before/after in ~10 lines of code.

### Using a Real VLA

```python
from isonome.core.config import AppConfig
from isonome.core.agent import Agent

config = AppConfig(
    agent_name="my_robot",
    jepa={"backend": "openvla", "model_id": "openvla/openvla-7b"},
    soma={"urdf_path": "my_robot.urdf"},
)
agent = Agent(config)
await agent.boot()
await agent.load_kernel()  # loads ~/.isonome/kernels/{robot_hash}.pt
await agent.run(duration_s=60.0)
```

## Calibration

The open-source **runtime** consumes calibrated kernels but does **not** produce them. The production calibration pipeline (auto-calibration engine, cloud training, certification logic) lives in a separate closed repository.

To use a calibrated kernel:

1. Train a kernel for your robot via the calibration pipeline (closed source).
2. Place the `.pt` file at `~/.isonome/kernels/{robot_hash}.pt`.
3. Call `agent.load_kernel()` at boot time.

Without a kernel, the framework falls back to `NaiveMapper` — a deterministic truncation/padding from canonical 14-DOF to your robot's N-DOF. This is safe but imprecise.

## Project Structure

```
my_robot/
  isonome.toml          # Framework manifest
  config.yaml           # Layer configs, VLA backend, safety thresholds
  main.py               # Entrypoint
  urdf/
    robot.urdf          # Body description
  kernels/
    {robot_hash}.pt     # Calibrated kernel (produced externally)
```

## Configuration

Edit `config.yaml` to configure the VLA backend, layer frequencies, and safety thresholds:

```yaml
jepa:
  backend: openvla
  model_id: openvla/openvla-7b

soma:
  urdf_path: urdf/robot.urdf

reflex:
  control_freq_hz: 100.0
  policy_freq_hz: 1.0
```

## License

MIT
