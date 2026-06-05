"""Mock simulation bridge for frontend development without Isaac Sim.

Provides the same WebSocket + MJPEG interface as IsaacSimBridge but simulates
a simple dummy robot in software. Useful for testing the dashboard UI when
Isaac Sim is not installed or available.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import math
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

try:
    import numpy as np
except ImportError:
    np = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

try:
    import websockets

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

logger = logging.getLogger("isonome.sim.mock_bridge")


class MockRobot:
    """A software-simulated robot with simple physics-based joint dynamics.

    Each joint is modeled as a damped pendulum with gravity-restoring torque.
    This produces physically plausible motion rather than synthetic sine waves.
    """

    def __init__(self, joint_names: list[str]) -> None:
        self.joint_names = joint_names
        n = len(joint_names)
        # Start from random initial displacements (as if dropped)
        self.positions = [0.0] * n
        self.velocities = [0.0] * n
        self.accelerations = [0.0] * n
        self.time = 0.0

        # Physics params per joint
        self._gravity = [4.0 + i * 0.5 for i in range(n)]   # restoring torque
        self._damping = [0.3 + i * 0.05 for i in range(n)]  # velocity damping
        self._mass = [1.0] * n
        self._length = [0.15 + i * 0.02 for i in range(n)]  # pendulum length

        # Perturb initial state so joints are not all at zero
        import random
        random.seed(42)
        for i in range(n):
            self.positions[i] = random.uniform(-0.6, 0.6)
            self.velocities[i] = random.uniform(-0.2, 0.2)

    def step(self, dt: float = 1.0 / 60.0) -> None:
        """Integrate one physics step using semi-implicit Euler."""
        self.time += dt
        for i, name in enumerate(self.joint_names):
            # Torque = -gravity * sin(pos) - damping * vel
            # Approx: for small angles, sin(pos) ~ pos, but use actual sin for realism
            torque = (
                -self._gravity[i] * math.sin(self.positions[i])
                - self._damping[i] * self.velocities[i]
            )
            # alpha = torque / (m * l^2)
            inertia = self._mass[i] * self._length[i] ** 2
            self.accelerations[i] = torque / inertia
            self.velocities[i] += self.accelerations[i] * dt
            self.positions[i] += self.velocities[i] * dt

            # Hard clamp to joint limits (-pi, pi)
            if self.positions[i] > math.pi:
                self.positions[i] = math.pi
                self.velocities[i] *= -0.3  # bounce
            elif self.positions[i] < -math.pi:
                self.positions[i] = -math.pi
                self.velocities[i] *= -0.3

    def get_state(self) -> dict[str, Any]:
        joints = []
        for i, name in enumerate(self.joint_names):
            joints.append(
                {
                    "name": name,
                    "position": round(self.positions[i], 4),
                    "velocity": round(self.velocities[i], 4),
                }
            )
        return {
            "joints": joints,
            "base_pose": {
                "position": [0.0, 0.0, 0.5],
                "orientation": [0.0, 0.0, 0.0, 1.0],
            },
            "playing": True,
            "timestamp": self.time,
        }


class MockSimBridge:
    """Mock bridge that mimics IsaacSimBridge's WebSocket + MJPEG interface."""

    def __init__(self, websocket_port: int = 8765, mjpeg_port: int = 8766) -> None:
        if not HAS_WEBSOCKETS:
            raise RuntimeError("websockets package is required. pip install websockets")
        self._ws_port = websocket_port
        self._mjpeg_port = mjpeg_port
        self._robot: MockRobot | None = None
        self._playing = False
        self._joint_names: list[str] = []
        self._mjpeg_clients: set[asyncio.StreamWriter] = set()
        self._frame_task: asyncio.Task | None = None
        self._sim_task: asyncio.Task | None = None
        self._server: Any = None
        self._mjpeg_server: asyncio.AbstractServer | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self._server = await websockets.serve(
            self._handle_client, "0.0.0.0", self._ws_port
        )
        self._mjpeg_server = await asyncio.start_server(
            self._handle_mjpeg_client, "0.0.0.0", self._mjpeg_port
        )
        logger.info(
            "MockSimBridge listening",
            extra={"websocket": self._ws_port, "mjpeg": self._mjpeg_port},
        )
        # Start sim tick loop
        self._sim_task = asyncio.create_task(self._sim_loop())
        await asyncio.Future()

    def shutdown(self) -> None:
        if self._server:
            self._server.close()
        if self._mjpeg_server:
            self._mjpeg_server.close()
        if self._frame_task:
            self._frame_task.cancel()
        if self._sim_task:
            self._sim_task.cancel()
        logger.info("MockSimBridge shut down")

    # ------------------------------------------------------------------
    # Sim loop
    # ------------------------------------------------------------------

    async def _sim_loop(self) -> None:
        while True:
            if self._playing and self._robot:
                self._robot.step()
            await asyncio.sleep(1.0 / 60.0)

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    async def _handle_client(self, websocket: Any) -> None:
        logger.info("Client connected", extra={"remote": websocket.remote_address})
        try:
            async for message in websocket:
                try:
                    cmd = json.loads(message)
                    response = await self._handle_command(cmd)
                except Exception as exc:
                    response = {"error": str(exc)}
                await websocket.send(json.dumps(response))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            logger.info("Client disconnected")

    async def _handle_command(self, cmd: dict[str, Any]) -> dict[str, Any]:
        action = cmd.get("action")

        if action == "load_urdf":
            return self._cmd_load_urdf(cmd.get("path", ""))
        if action == "play":
            self._playing = True
            return {"ok": True, "playing": True}
        if action == "pause":
            self._playing = False
            return {"ok": True, "playing": False}
        if action == "step":
            steps = cmd.get("steps", 1)
            if self._robot:
                for _ in range(steps):
                    self._robot.step()
            return {"ok": True, "state": self._get_state_dict()}
        if action == "reset":
            if self._robot:
                self._robot.positions = [0.0] * len(self._joint_names)
                self._robot.velocities = [0.0] * len(self._joint_names)
                self._robot.time = 0.0
            return {"ok": True}
        if action == "get_state":
            return {"ok": True, "state": self._get_state_dict()}
        if action == "set_joints":
            positions = cmd.get("positions", {})
            if self._robot:
                for name, pos in positions.items():
                    if name in self._robot.joint_names:
                        idx = self._robot.joint_names.index(name)
                        self._robot.positions[idx] = pos
            return {"ok": True}

        return {"error": f"unknown action: {action}"}

    def _cmd_load_urdf(self, path: str) -> dict[str, Any]:
        urdf_path = Path(path)
        if not urdf_path.exists():
            return {"error": f"URDF not found: {path}"}

        # Parse URDF to extract joint names
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        joint_names = []
        for joint in root.findall("joint"):
            jtype = joint.get("type", "fixed")
            if jtype != "fixed":
                name = joint.get("name", f"joint_{len(joint_names)}")
                joint_names.append(name)

        self._joint_names = joint_names
        self._robot = MockRobot(joint_names)
        self._playing = True

        return {
            "ok": True,
            "joints": joint_names,
            "dof_count": len(joint_names),
        }

    def _get_state_dict(self) -> dict[str, Any]:
        if self._robot:
            return self._robot.get_state()
        return {
            "joints": [],
            "base_pose": None,
            "playing": self._playing,
            "timestamp": 0.0,
        }

    # ------------------------------------------------------------------
    # MJPEG
    # ------------------------------------------------------------------

    async def _handle_mjpeg_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._mjpeg_clients.add(writer)
        logger.info("MJPEG client connected")

        headers = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n\r\n"
        )
        writer.write(headers.encode())
        await writer.drain()

        try:
            while writer in self._mjpeg_clients:
                frame = self._render_mock_frame()
                if frame:
                    boundary = b"--frame\r\n"
                    ct = b"Content-Type: image/jpeg\r\n"
                    cl = f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    writer.write(boundary + ct + cl + frame + b"\r\n")
                    await writer.drain()
                await asyncio.sleep(1.0 / 30.0)
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self._mjpeg_clients.discard(writer)
            writer.close()
            logger.info("MJPEG client disconnected")

    def _render_mock_frame(self) -> bytes:
        """Render a synthetic viewport image showing robot state.

        Industrial monochrome aesthetic. No gradients, no rounded shapes.
        """
        if Image is None:
            return b""
        width, height = 640, 480
        img = Image.new("RGB", (width, height), color=(10, 10, 10))
        draw = ImageDraw.Draw(img)

        # Grid lines
        for gx in range(0, width, 40):
            draw.line([(gx, 0), (gx, height)], fill=(20, 20, 20), width=1)
        for gy in range(0, height, 40):
            draw.line([(0, gy), (width, gy)], fill=(20, 20, 20), width=1)

        # Header bar
        draw.rectangle([0, 0, width, 28], fill=(18, 18, 18))
        draw.line([(0, 28), (width, 28)], fill=(50, 50, 50), width=1)
        draw.text((8, 6), "ISONOME SIM SCAFFOLD / MOCK PHYSICS", fill=(150, 150, 150))
        draw.text((width - 180, 6), f"JOINTS: {len(self._joint_names)}  PLAYING: {self._playing}", fill=(150, 150, 150))

        if self._robot:
            # Joint state table on left
            table_x = 20
            table_y = 44
            row_h = 18
            col_w_name = 80
            col_w_bar = 140
            col_w_val = 60

            # Header
            draw.text((table_x, table_y), "JOINT", fill=(100, 100, 100))
            draw.text((table_x + col_w_name + 4, table_y), "POSITION", fill=(100, 100, 100))
            draw.text((table_x + col_w_name + col_w_bar + 8, table_y), "VALUE", fill=(100, 100, 100))
            draw.line([
                (table_x, table_y + 14),
                (table_x + col_w_name + col_w_bar + col_w_val, table_y + 14)
            ], fill=(50, 50, 50), width=1)

            for i, joint in enumerate(self._robot.get_state()["joints"]):
                y = table_y + 20 + i * row_h
                name = joint["name"][:10]
                pos = joint["position"]
                vel = joint["velocity"]

                # Name
                draw.text((table_x, y), name, fill=(180, 180, 180))

                # Bar background
                bar_x = table_x + col_w_name + 4
                draw.rectangle(
                    [bar_x, y + 2, bar_x + col_w_bar, y + 12],
                    fill=(8, 8, 8),
                    outline=(40, 40, 40),
                )
                # Bar fill (cyan for position)
                fill_w = int((pos + math.pi) / (2 * math.pi) * col_w_bar)
                fill_w = max(0, min(col_w_bar, fill_w))
                if fill_w > 0:
                    draw.rectangle(
                        [bar_x, y + 2, bar_x + fill_w, y + 12],
                        fill=(0, 140, 140),
                    )
                # Zero line
                zero_x = bar_x + col_w_bar // 2
                draw.line([(zero_x, y + 2), (zero_x, y + 12)], fill=(80, 80, 80), width=1)

                # Value
                draw.text((bar_x + col_w_bar + 8, y), f"{pos:+.3f}", fill=(200, 200, 200))

            # Kinematic schematic on right
            sc_x = 360
            sc_y = 80
            sc_w = 260
            sc_h = 360
            draw.rectangle([sc_x, sc_y, sc_x + sc_w, sc_y + sc_h], outline=(30, 30, 30), width=1)
            draw.text((sc_x + 6, sc_y + 4), "KINEMATIC VIEW", fill=(100, 100, 100))

            angles = [j["position"] for j in self._robot.get_state()["joints"]]
            if len(angles) >= 2:
                cx = sc_x + sc_w // 2
                cy = sc_y + sc_h - 40
                link_len = 35
                x, y = cx, cy
                base_angle = -math.pi / 2
                for i, ang in enumerate(angles[:8]):
                    base_angle += ang * 0.4
                    nx = x + link_len * math.cos(base_angle)
                    ny = y + link_len * math.sin(base_angle)
                    draw.line([(x, y), (nx, ny)], fill=(0, 170, 170), width=2)
                    draw.rectangle([x - 2, y - 2, x + 2, y + 2], fill=(200, 200, 200))
                    x, y = nx, ny
                draw.rectangle([x - 3, y - 3, x + 3, y + 3], fill=(200, 50, 50))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Isonome Mock Sim Bridge (dev mode)")
    parser.add_argument("--ws-port", type=int, default=8765)
    parser.add_argument("--mjpeg-port", type=int, default=8766)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    bridge = MockSimBridge(
        websocket_port=args.ws_port,
        mjpeg_port=args.mjpeg_port,
    )
    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
