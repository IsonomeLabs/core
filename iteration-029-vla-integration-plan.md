# Iteration 029: VLA Integration Plan (π0.5 / OpenVLA)

> Status: Planned  
> Goal: Close the loop between MuJoCo physics simulation and a Vision-Language-Action (VLA) model so the agent observes, reasons, and acts in real time.

---

## 1. Observation Pipeline

**Current state:** The MuJoCo bridge renders a viewport and streams joint states via WebSocket.

**Gap:** VLA models need structured observations — an image plus proprioception plus a task description.

### Bridge Hook

Add `get_observation()` to `MuJoCoBridge` in `isonome/sim/mujoco_bridge.py`:

```python
def get_observation(self, intent: str = "") -> dict:
    """Package current sim state into a VLA observation."""
    return {
        "image": self._capture_frame_rgb(),           # [H, W, 3] uint8
        "proprioception": self._get_proprio(),        # [n_joints * 2] pos + vel
        "intent": intent,
        "timestamp": time.time(),
    }

def _get_proprio(self) -> np.ndarray:
    """Normalized joint positions and velocities."""
    # Fill with current qpos / qvel, normalized per joint limit
    ...
```

---

## 2. Action-Space Mapping

**Current state:** Bridge accepts `set_joints` (absolute position targets).

**Gap:** VLA models output actions in their own action space — typically delta positions or joint velocities.

### Mapping Layer

```python
def vla_action_to_ctrl(action: np.ndarray) -> dict:
    """
    Convert VLA output to MuJoCo control.

    action shape: [n_joints] — predicted delta positions or velocities
    Returns: {joint_name: target_value}
    """
    # Option A: Delta positions
    for i, name in enumerate(self._joint_names):
        current = self._data.qpos[self._model.jnt_qposadr[i]]
        self._data.ctrl[i] = current + action[i]

    # Option B: Action chunking — VLA predicts T future steps
    self._action_queue.extend(action_chunk)
```

Add action smoothing:
```python
self._action_lpf_alpha = 0.3  # low-pass filter coefficient
```

---

## 3. Intent Interface

### New WebSocket Command

```json
{"action": "set_intent", "text": "reach the red cube"}
```

- Stores `self._current_intent` in `MuJoCoBridge`
- Sent to VLA as text conditioning at every inference step
- UI: text input in `dashboard/sim.html` toolbar

### Dashboard UI Addition

```html
<div class="toolbar-group">
  <input type="text" id="intent-input" placeholder="Intent…" class="btn" style="width:200px">
  <button id="btn-intent" class="btn primary">SET INTENT</button>
</div>
```

---

## 4. Model Loading Strategy

| Option | Model | Params | GPU | Notes |
|---|---|---|---|---|
| A | π0.5 | ~7B | A100/H100 | Best performance, closed weights |
| B | OpenVLA | ~7B | 24GB VRAM | Open weights, good baseline |
| C | LLaVA-robot | ~7B | 24GB VRAM | Lightweight, faster iteration |

**Recommendation:** Start with **Option C (LLaVA-robot)** or **B (OpenVLA)** for fast iteration. Swap to π0.5 for production runs.

### Wrapper Structure

```
isonome/praxis/vla/
├── __init__.py
├── base.py          # VLABase abstract class
├── openvla.py       # OpenVLA wrapper
├── llava_robot.py   # LLaVA-robot wrapper
└── pi05.py          # π0.5 wrapper (placeholder)
```

```python
class VLABase:
    def load(self, checkpoint_path: str) -> None: ...
    def predict(self, obs: dict) -> np.ndarray: ...
    def train_step(self, batch: dict) -> dict: ...
```

---

## 5. Training Loop

### Data Collection

Every timestep, store to replay buffer:

```python
{
    "obs": { "image": ..., "proprioception": ..., "intent": ... },
    "action": action_applied,
    "reward": reward,
    "done": done,
}
```

**Reward options:**
- **Sparse:** `+1` on task completion, `0` otherwise
- **Dense:** Distance to target (from MuJoCo body positions), velocity penalty, etc.

### Training Modes

| Mode | Data | Algorithm | Use Case |
|---|---|---|---|
| Offline | Collected episodes | Behavior cloning + intent conditioning | Bootstrap policy |
| Online | Interleaved | DAgger / RL (PPO, SAC) | Fine-tune in sim |

### Infrastructure

```
examples/train_vla.py       # Training script
examples/collect_demo.py    # Human tele-op data collection
```

Log to Weights & Biases:
```python
wandb.log({
    "loss": loss.item(),
    "reward_mean": rewards.mean(),
    "success_rate": success_rate,
})
```

Save checkpoints as `.pt` or Safetensors every N steps.

---

## 6. Closed-Loop Control Flow

```
┌─────────────────┐      image + proprio + intent       ┌─────────────┐
│  MuJoCo Bridge  │ ───────────────────────────────────→ │  VLA Model  │
│  (60 Hz physics)│                                      │  (10 Hz)    │
└─────────────────┘                                      └─────────────┘
         ↑                                                      │
         │              action chunk [T x n_joints]             │
         └──────────────────────────────────────────────────────┘
```

1. Bridge renders viewport, reads joint state
2. Observation packaged → sent to VLA
3. VLA predicts action chunk (e.g., 8 future steps)
4. Bridge applies first action, queues remainder
5. MuJoCo steps physics at 60 Hz
6. Loop

---

## 7. First Milestone: "Reach Target"

A simple Cartesian reaching task to validate the full pipeline.

### Task Definition

User types intent:
```
"reach x=0.5 y=0.2 z=0.3"
```

Or selects a target visually:
```
"reach the red cube"
```

### Success Criteria

- End-effector position within 5 cm of target
- Smooth motion (no jerky motor commands)
- Completes in < 5 seconds

### Validation Steps

1. Load `examples/robot_arm.xml`
2. Set intent via dashboard
3. VLA predicts joint deltas
4. Bridge applies actions
5. MuJoCo simulates physics
6. Repeat at 10 Hz inference / 60 Hz physics
7. Log trajectory for analysis

---

## 8. Files to Touch

| File | Change |
|---|---|
| `isonome/sim/mujoco_bridge.py` | Add `get_observation()`, `_get_proprio()`, `set_intent` handler |
| `isonome/praxis/vla/__init__.py` | New package |
| `isonome/praxis/vla/base.py` | `VLABase` abstract class |
| `isonome/praxis/vla/openvla.py` | OpenVLA wrapper |
| `dashboard/isaac_client.js` | Add intent input + `set_intent` command |
| `dashboard/sim.html` | Intent text box in toolbar |
| `examples/train_vla.py` | Training script |
| `examples/collect_demo.py` | Tele-op demo collection |

---

## 9. Dependencies to Add

```bash
pip install transformers accelerate bitsandbytes einops
# Optional:
pip install wandb opencv-python
```

---

## Open Questions

1. **Action representation:** Delta positions vs. velocity vs. end-effector pose?
2. **Camera view:** Single fixed camera or multiple views?
3. **Real-world transfer:** Train in sim, deploy on real robot via same bridge interface?
4. **Safety:** Joint limit clamps, velocity limits, emergency stop via dashboard?
