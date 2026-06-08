"""VLA Steering — full closed-loop demo with LLM → VLA → MuJoCo.

Architecture
------------
    User command ──→ MockLLMSteering ──→ VLA intent
                          ↓
                    Quadcamera obs ──→ MockVLABackend ──→ action
                          ↓
                    VLAController ──→ MuJoCo physics
                          ↓
                    VLAInspector  ──→ terminal + dashboard panel

Usage
-----
    # Headless demo
    python examples/vla_steering.py

    # With dashboard (open http://localhost:8765/sim.html)
    python examples/vla_steering.py --serve

    # Custom command
    python examples/vla_steering.py --command "reach the red cube"
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from isonome.praxis.vla import MockVLABackend
from isonome.sim.llm_steering import MockLLMSteering
from isonome.sim.mujoco_bridge import MuJoCoBridge
from isonome.sim.vla_controller import VLAController
from isonome.sim.vla_inspector import VLAInspector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vla_steering")

# Target joint configuration for the dummy arm reach pose
REACH_TARGET_QPOS = np.array([0.5, 0.6, -0.2, -0.3, 0.3, 0.0, 0.0], dtype=np.float32)


def build_scene_xml(target_pos: tuple[float, float, float] = (0.40, 0.40, 1.30)) -> str:
    """Return a MuJoCo XML string with the arm and a red cube target."""
    tx, ty, tz = target_pos
    return f"""<mujoco model="reach_scene">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0" integrator="RK4"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.15 0.15 0.2" rgb2="0.25 0.25 0.35" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.25 0.3" rgb2="0.3 0.3 0.35" width="512" height="512"/>
    <material name="grid_mat" texture="grid" texrepeat="8 8" reflectance="0.1"/>
    <material name="base_mat" rgba="0.6 0.6 0.65 1"/>
    <material name="link_mat" rgba="0.1 0.85 0.9 1"/>
    <material name="end_mat" rgba="0.95 0.35 0.35 1"/>
    <material name="target_mat" rgba="0.95 0.2 0.2 0.8"/>
  </asset>

  <worldbody>
    <light directional="true" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" pos="2 2 5" dir="-0.3 -0.3 -0.8"/>
    <light directional="true" diffuse="0.5 0.5 0.55" specular="0.15 0.15 0.15" pos="-2 -2 4" dir="0.25 0.25 -0.8"/>

    <geom name="floor" type="plane" size="10 10 0.1" material="grid_mat" pos="0 0 -0.01"/>

    <body name="target" pos="{tx} {ty} {tz}">
      <geom name="target_geom" type="box" size="0.04 0.04 0.04" material="target_mat"/>
      <site name="target_site" pos="0 0 0" size="0.02"/>
    </body>

    <body name="base_link" pos="0 0 0">
      <geom name="base_geom" type="box" size="0.12 0.12 0.06" material="base_mat"/>

      <body name="l1" pos="0 0 0.15">
        <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="l1_geom" type="capsule" size="0.05" fromto="0 0 0 0 0 0.3" material="link_mat"/>

        <body name="l2" pos="0 0 0.3">
          <joint name="j2" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
          <geom name="l2_geom" type="capsule" size="0.045" fromto="0 0 0 0 0 0.3" material="link_mat"/>

          <body name="l3" pos="0 0 0.3">
            <joint name="j3" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
            <geom name="l3_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.24" material="link_mat"/>

            <body name="l4" pos="0 0 0.24">
              <joint name="j4" type="hinge" axis="1 0 0" range="-1.5 1.5"/>
              <geom name="l4_geom" type="capsule" size="0.035" fromto="0 0 0 0 0 0.2" material="link_mat"/>

              <body name="l5" pos="0 0 0.2">
                <joint name="j5" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                <geom name="l5_geom" type="capsule" size="0.03" fromto="0 0 0 0 0 0.16" material="link_mat"/>

                <body name="l6" pos="0 0 0.16">
                  <joint name="j6" type="hinge" axis="1 0 0" range="-1.5 1.5"/>
                  <geom name="l6_geom" type="capsule" size="0.025" fromto="0 0 0 0 0 0.12" material="link_mat"/>

                  <body name="l7" pos="0 0 0.12">
                    <joint name="j7" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                    <geom name="l7_geom" type="capsule" size="0.022" fromto="0 0 0 0 0 0.08" material="end_mat"/>
                    <site name="ee" pos="0 0 0.08" size="0.02"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def compute_reach_error(ee_pos: np.ndarray | None, target_pos: np.ndarray) -> float:
    if ee_pos is None:
        return float("inf")
    return float(np.linalg.norm(ee_pos - target_pos))


def print_header() -> None:
    print("=" * 80)
    print("  VLA STEERING DEMO — Iteration 029")
    print("  Quadcameral Architecture: Soma → JEPA → Cortex → Reflex")
    print("  Human → LLM (Cortex) → VLA (JEPA) → MuJoCo (Soma+Reflex)")
    print("=" * 80)
    print()


def print_vla_panel(inspector: VLAInspector, latest: dict[str, Any] | None) -> None:
    """Print a terminal panel showing the last VLA I/O."""
    print("─" * 80)
    print("  VLA INSPECTOR PANEL")
    print("─" * 80)
    if latest:
        print(f"  Step:      {latest['step']}")
        print(f"  Time:      {latest['timestamp']}")
        print(f"  Intent:    '{latest['intent']}'")
        print(f"  Cameras:   {latest['n_cameras']}")
        print(f"  Proprio:   {latest['proprio_shape']}")
        print(f"  Action:    shape={latest['action_shape']} norm={latest['action_norm']:.4f}")
        print(f"  Action[:4]: {latest['action_head']}")
        if latest.get("ee_pos"):
            print(f"  EE pos:    [{latest['ee_pos'][0]:+.3f}, {latest['ee_pos'][1]:+.3f}, {latest['ee_pos'][2]:+.3f}]")
    print("─" * 80)


async def main() -> None:
    parser = argparse.ArgumentParser(description="VLA Steering Demo")
    parser.add_argument("--serve", action="store_true", help="Start WebSocket/MJPEG servers")
    parser.add_argument("--steps", type=int, default=150, help="Inference steps")
    parser.add_argument("--command", type=str, default="reach", help="User command")
    parser.add_argument("--target", type=float, nargs=3, default=[0.40, 0.40, 1.30], metavar="X Y Z")
    parser.add_argument("--inference_hz", type=float, default=10.0)
    parser.add_argument("--physics_hz", type=float, default=60.0)
    parser.add_argument("--cameras", type=str, default="quad", choices=["tracking", "quad"])
    args = parser.parse_args()

    target_pos = np.array(args.target, dtype=np.float32)
    print_header()

    # 1. Build scene
    xml_path = Path(tempfile.gettempdir()) / "isonome_reach_scene.xml"
    xml_path.write_text(build_scene_xml(tuple(args.target)))

    # 2. Load bridge
    bridge = MuJoCoBridge(websocket_port=8765, mjpeg_port=8766)
    result = bridge._cmd_load_urdf(str(xml_path))
    if not result.get("ok"):
        logger.error("Failed to load scene: %s", result.get("error"))
        return
    n_joints = len(bridge._joint_names)
    logger.info("Loaded %d joints: %s", n_joints, bridge._joint_names)

    # 3. Reset to zero pose (stable start)
    if bridge._model is not None and bridge._data is not None:
        import mujoco
        mujoco.mj_resetData(bridge._model, bridge._data)
        mujoco.mj_forward(bridge._model, bridge._data)

    # 4. Cortex — LLM steering (translates user command → VLA intent)
    steering = MockLLMSteering()
    intent = await steering.generate_intent(args.command)
    print(f"User command: '{args.command}'")
    print(f"Cortex/LLM:   '{intent}'")
    print()

    # 5. JEPA — frozen VLA policy (deliberates action from RAW observation)
    vla = MockVLABackend(target=REACH_TARGET_QPOS[:n_joints], action_dim=n_joints)

    # 6. Inspector — visualize VLA I/O in real time
    inspector = VLAInspector(max_entries=200)

    # 7. Soma + Reflex — controller applies actions with safety smoothing
    controller = VLAController(bridge, vla, intent=intent, action_lpf_alpha=0.3)

    # 8. Optionally start servers for remote dashboard viewing
    loop = asyncio.get_event_loop()
    if args.serve:
        logger.info("Starting servers on ws=8765 mjpeg=8766")
        loop.create_task(bridge.run())
        print("Dashboard: http://localhost:8765/sim.html")
        print()

    # 9. Closed-loop tick (perceive → deliberate → act → observe)
    physics_substeps = max(1, int(round(args.physics_hz / args.inference_hz)))
    dt_inference = 1.0 / args.inference_hz

    print(f"Running {args.steps} ticks @ {args.inference_hz} Hz ({physics_substeps} physics substeps)")
    print()

    for i in range(args.steps):
        t0 = time.perf_counter()

        # Observe with quadcamera
        obs = bridge.get_observation(intent, cameras=args.cameras)

        # Predict
        action = vla.predict(obs)
        if action.ndim == 2:
            action = action[0]
        action = np.asarray(action, dtype=np.float32)

        # Smooth
        if controller._last_action is not None and action.shape == controller._last_action.shape:
            action = controller._action_lpf_alpha * action + (1.0 - controller._action_lpf_alpha) * controller._last_action
        controller._last_action = action.copy()

        # Act — pure kinematics (gravity=0 in this scene)
        controller._apply_action(action)

        # Inspect
        ee_pos = controller._end_effector_pos()
        entry = inspector.log_step(obs, action, ee_pos.tolist() if ee_pos is not None else None)

        # Print panel every 10 steps and at the end
        if i % 10 == 0 or i == args.steps - 1:
            print_vla_panel(inspector, entry)

        # Throttle (async so bridge servers get event-loop time)
        elapsed = time.perf_counter() - t0
        sleep = dt_inference - elapsed
        if sleep > 0:
            await asyncio.sleep(sleep)

    # 10. Summary
    start_ee = np.array(inspector._buffer[0]["ee_pos"], dtype=np.float32) if inspector._buffer[0].get("ee_pos") else None
    end_ee = np.array(inspector._buffer[-1]["ee_pos"], dtype=np.float32) if inspector._buffer[-1].get("ee_pos") else None
    start_err = compute_reach_error(start_ee, target_pos)
    end_err = compute_reach_error(end_ee, target_pos)

    print()
    print("=" * 80)
    print("  RESULTS")
    print("=" * 80)
    if start_ee is not None and end_ee is not None:
        print(f"  Start EE:   [{start_ee[0]:+.3f}, {start_ee[1]:+.3f}, {start_ee[2]:+.3f}]")
        print(f"  End EE:     [{end_ee[0]:+.3f}, {end_ee[1]:+.3f}, {end_ee[2]:+.3f}]")
    print(f"  Start dist: {start_err:.4f} m")
    print(f"  End dist:   {end_err:.4f} m")
    print(f"  Steps:      {args.steps}")
    print(f"  Camera mode:{args.cameras}")
    print("=" * 80)

    if args.serve:
        # Give the dashboard a moment to receive the final frames, then exit
        await asyncio.sleep(2.0)
        bridge.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
