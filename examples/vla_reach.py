"""VLA Reach — closed-loop demo for Iteration 029.

Loads the 7-DOF dummy arm, adds a red cube target in the scene, and lets a
Mock VLA backend drive the arm toward a reaching pose.  The controller runs
entirely in Python; the WebSocket / MJPEG servers are optional and only
needed if you want to watch from the dashboard.

Usage
-----
    # 1. Run with live dashboard viewing (optional)
    python examples/vla_reach.py --serve

    # 2. Headless — just print trajectory stats
    python examples/vla_reach.py
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import tempfile
from pathlib import Path

import numpy as np

from isonome.praxis.vla import MockVLABackend
from isonome.sim.mujoco_bridge import MuJoCoBridge
from isonome.sim.vla_controller import VLAController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vla_reach")

# Target joint configuration that puts the end-effector roughly at
# x≈0.42, y≈0.41, z≈1.36 (a natural reach pose for the dummy arm)
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

    <!-- Target cube -->
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
          <joint name="j2" type="hinge" axis="0 1 0" range="-2.0 2.0" />
          <geom name="l2_geom" type="capsule" size="0.045" fromto="0 0 0 0 0 0.3" material="link_mat"/>

          <body name="l3" pos="0 0 0.3">
            <joint name="j3" type="hinge" axis="0 1 0" range="-2.0 2.0" />
            <geom name="l3_geom" type="capsule" size="0.04" fromto="0 0 0 0 0 0.24" material="link_mat"/>

            <body name="l4" pos="0 0 0.24">
              <joint name="j4" type="hinge" axis="1 0 0" range="-1.5 1.5" />
              <geom name="l4_geom" type="capsule" size="0.035" fromto="0 0 0 0 0 0.2" material="link_mat"/>

              <body name="l5" pos="0 0 0.2">
                <joint name="j5" type="hinge" axis="0 1 0" range="-1.5 1.5" />
                <geom name="l5_geom" type="capsule" size="0.03" fromto="0 0 0 0 0 0.16" material="link_mat"/>

                <body name="l6" pos="0 0 0.16">
                  <joint name="j6" type="hinge" axis="1 0 0" range="-1.5 1.5" />
                  <geom name="l6_geom" type="capsule" size="0.025" fromto="0 0 0 0 0 0.12" material="link_mat"/>

                  <body name="l7" pos="0 0 0.12">
                    <joint name="j7" type="hinge" axis="0 1 0" range="-1.5 1.5" />
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
    """Cartesian distance between end-effector and target."""
    if ee_pos is None:
        return float("inf")
    return float(np.linalg.norm(ee_pos - target_pos))


def main() -> None:
    parser = argparse.ArgumentParser(description="VLA Reach Demo")
    parser.add_argument("--serve", action="store_true", help="Start WebSocket/MJPEG servers for dashboard viewing")
    parser.add_argument("--steps", type=int, default=300, help="Number of inference steps")
    parser.add_argument("--inference_hz", type=float, default=10.0)
    parser.add_argument("--physics_hz", type=float, default=60.0)
    parser.add_argument("--target", type=float, nargs=3, default=[0.40, 0.40, 1.30], metavar="X Y Z")
    args = parser.parse_args()

    target_pos = np.array(args.target, dtype=np.float32)

    # Write temporary XML with the target cube
    xml_path = Path(tempfile.gettempdir()) / "isonome_reach_scene.xml"
    xml_path.write_text(build_scene_xml(tuple(args.target)))

    # 1. Load bridge
    bridge = MuJoCoBridge(websocket_port=8765, mjpeg_port=8766)
    result = bridge._cmd_load_urdf(str(xml_path))
    if not result.get("ok"):
        logger.error("Failed to load scene: %s", result.get("error"))
        return
    n_joints = len(bridge._joint_names)
    logger.info("Loaded %d joints: %s", n_joints, bridge._joint_names)

    # 2. Create VLA model with a reach target in joint space
    vla = MockVLABackend(target=REACH_TARGET_QPOS[:n_joints], action_dim=n_joints)

    # 3. Create controller
    controller = VLAController(
        bridge,
        vla,
        intent="reach the target",
        action_lpf_alpha=0.3,
    )

    # 5. Run closed-loop
    print(f"\n=== VLA Reach Demo ===")
    print(f"Target:     x={target_pos[0]:.2f} y={target_pos[1]:.2f} z={target_pos[2]:.2f}")
    print(f"Joints:     {n_joints}")
    print(f"Steps:      {args.steps} @ {args.inference_hz} Hz")
    print(f"Physics:    {args.physics_hz} Hz")
    print()

    if args.serve:
        print("Dashboard: http://localhost:8420/sim")
        print()
        # Run controller in a background thread so the event loop stays free
        # for WebSocket / MJPEG serving
        async def _run_with_servers() -> None:
            logger.info("Starting servers on ws=8765 mjpeg=8766")
            await bridge.run()

        async def _run_controller() -> None:
            loop = asyncio.get_running_loop()
            trajectory = await loop.run_in_executor(
                None, controller.run, args.steps, args.inference_hz, args.physics_hz
            )
            # Report results
            start_ee = np.array(trajectory[0]["ee_pos"], dtype=np.float32)
            end_ee = np.array(trajectory[-1]["ee_pos"], dtype=np.float32)
            start_err = compute_reach_error(start_ee, target_pos)
            end_err = compute_reach_error(end_ee, target_pos)
            print(f"Start EE:   [{start_ee[0]:+.3f}, {start_ee[1]:+.3f}, {start_ee[2]:+.3f}]")
            print(f"End EE:     [{end_ee[0]:+.3f}, {end_ee[1]:+.3f}, {end_ee[2]:+.3f}]")
            print(f"Start dist: {start_err:.4f} m")
            print(f"End dist:   {end_err:.4f} m")
            print(f"Improvement: {start_err - end_err:.4f} m")
            if end_err < 0.05:
                print("\n✓ Success — end-effector within 5 cm of target")
            elif end_err < 0.15:
                print("\n~ Partial — within 15 cm, more tuning needed")
            else:
                print("\n✗ Miss — target not reached")
            print("\nSimulation complete. Servers still running. Press Ctrl-C to exit.")
            await asyncio.Future()  # keep alive

        async def _main_serve() -> None:
            await asyncio.gather(_run_with_servers(), _run_controller())

        try:
            asyncio.run(_main_serve())
        except (KeyboardInterrupt, asyncio.CancelledError):
            bridge.shutdown()
        return

    trajectory = controller.run(
        n_steps=args.steps,
        inference_hz=args.inference_hz,
        physics_hz=args.physics_hz,
    )

    # 6. Report
    start_ee = np.array(trajectory[0]["ee_pos"], dtype=np.float32)
    end_ee = np.array(trajectory[-1]["ee_pos"], dtype=np.float32)
    start_err = compute_reach_error(start_ee, target_pos)
    end_err = compute_reach_error(end_ee, target_pos)

    print(f"Start EE:   [{start_ee[0]:+.3f}, {start_ee[1]:+.3f}, {start_ee[2]:+.3f}]")
    print(f"End EE:     [{end_ee[0]:+.3f}, {end_ee[1]:+.3f}, {end_ee[2]:+.3f}]")
    print(f"Start dist: {start_err:.4f} m")
    print(f"End dist:   {end_err:.4f} m")
    print(f"Improvement: {start_err - end_err:.4f} m")

    if end_err < 0.05:
        print("\n✓ Success — end-effector within 5 cm of target")
    elif end_err < 0.15:
        print("\n~ Partial — within 15 cm, more tuning needed")
    else:
        print("\n✗ Miss — target not reached")


if __name__ == "__main__":
    main()
