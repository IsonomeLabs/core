# Isonome

> Agentic robotics framework with hierarchical cognitive architecture.

Isonome is an open-source framework for building embodied AI systems. The core thesis: embodied intelligence isn't a monolithic LLM -- it's four parallel and cascading layers of cognition, from millisecond reflexes to long-term structural adaptation.

## The Four Layers

| Layer | Frequency | Role | LLM? |
|-------|-----------|------|------|
| **Reflex** | ~100Hz | Reactive motor control, collision avoidance, balance | No |
| **JEPA** | ~10Hz | Predictive world model, modulates Reflex | No |
| **Cortex** | ~0.5Hz | LLM-driven deliberation, advises JEPA | Yes |
| **Plasticity** | On-demand | LLM swarm rewrites kernels/tunes hyperparams | Yes |

### Control Flow

```
Sensor Input --+--> Reflex --> Motor Output
               |         ^
               +--> JEPA -+ (modulates Reflex)
                     ^
               Cortex -+ (advises JEPA only, never touches motors)

               Plasticity --> (rewrites all layers, gated by SafetyGovernor)
```

## Install

```bash
pip install isonome
```

With simulation support:

```bash
pip install isonome[sim]
```

## Quick Start

```bash
# Scaffold a new robot project
isonome init my-robot
cd my-robot

# Run in simulation
isonome sim

# Run on hardware
isonome run
```

## Safety Rules

The framework enforces strict safety rules for the Plasticity layer:

1. **Default**: Only runs when the robot is powered off or in guaranteed idle/safe state
2. **Boot mode**: Can run during boot if user explicitly sets `permit_boot_adaptation: true`
3. **Live mode**: Only triggered when the same failure occurs N or more times within a time window AND the robot is in SAFE_STATIONARY state
4. **Emergency stop**: Bypasses all layers and zeros motors immediately

All patches are transactional: the framework snapshots layer state, applies the patch, validates in simulation for 100 ticks before committing.

## Project Structure

```
my_robot/
  isonome.toml          # Framework manifest
  config.yaml           # Layer configs, LLM providers, safety thresholds
  main.py               # Entrypoint
  layers/
    reflex.py           # Your Reflex implementation
    jepa.py             # Your JEPA world model
    cortex.py           # Optional: custom deliberation hooks
    plasticity.py       # Optional: custom adaptation triggers
  sim/
    world.json          # Simulation world definition
  tests/
    test_agent.py
```

## Configuration

Edit `config.yaml` to configure layer frequencies, LLM providers, and safety thresholds:

```yaml
reflex:
  frequency_hz: 100.0
  max_latency_ms: 10.0

jepa:
  frequency_hz: 10.0
  prediction_horizon_s: 1.0

cortex:
  frequency_hz: 0.5
  provider: openai
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY

safety:
  permit_boot_adaptation: false
  error_repeat_threshold: 3
  error_window_s: 300

sim:
  engine: pybullet
  gui: false
```

## Built-in Presets

- **pet**: Reactive companion robot (fast reflexes, minimal deliberation)
- **patrol**: Autonomous patrol robot (balanced cognition, JEPA-heavy navigation)

## License

MIT
