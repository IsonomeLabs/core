"""Isaac Sim bridge for remote URDF simulation.

Run this script inside Isaac Sim's Python environment:

    cd /path/to/isaac-sim
    ./python.sh path/to/isonome/sim/isaac_bridge.py --port 8765

Or load as an Omniverse extension by copying `isaac_extension/` to your
Isaac Sim extensions folder.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any

# Isaac Sim imports are guarded — this module is designed to run inside
# Isaac Sim's Python environment where these packages are available.
try:
    import omni
    from omni.isaac.core import World
    from omni.isaac.core.utils.stage import open_stage
    from omni.isaac.urdf import _urdf
    from omni.kit.viewport.utility import get_active_viewport
    from pxr import Usd, UsdGeom

    HAS_ISAAC = True
except ImportError:
    HAS_ISAAC = False

try:
    import websockets

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    import numpy as np
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from av import VideoFrame

    HAS_AIORRTC = True
except ImportError:
    HAS_AIORRTC = False


logger = logging.getLogger("isonome.sim.isaac_bridge")


# ── WebRTC helpers ────────────────────────────────────────────────

class _SimVideoTrack(VideoStreamTrack):
    """Push frames from Isaac Sim viewport into a WebRTC video track."""

    kind = "video"

    def __init__(self, frame_source: Any) -> None:
        super().__init__()
        self._frame_source = frame_source  # callable -> np.ndarray RGB or None

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        arr = self._frame_source()
        if arr is None:
            arr = np.zeros((480, 640, 3), dtype=np.uint8)
        frame = VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


class WebRTCManager:
    """Manages PeerConnection lifecycle, signaling, and tracks."""

    def __init__(self, frame_source: Any, on_command: Any) -> None:
        if not HAS_AIORRTC:
            raise RuntimeError("aiortc is required: pip install aiortc")
        self._frame_source = frame_source
        self._on_command = on_command
        self._pcs: set[RTCPeerConnection] = set()
        self._channels: list[Any] = []
        self._track: _SimVideoTrack | None = None

    def _ensure_track(self) -> _SimVideoTrack:
        if self._track is None:
            self._track = _SimVideoTrack(self._frame_source)
        return self._track

    async def handle_offer(self, sdp: str, type_: str = "offer") -> dict[str, str]:
        pc = RTCPeerConnection(
            configuration={
                "iceServers": [{"urls": "stun:stun.l.google.com:19302"}]
            }
        )
        self._pcs.add(pc)

        @pc.on("connectionstatechange")
        async def _on_state_change() -> None:
            logger.info("PC state: %s", pc.connectionState)
            if pc.connectionState in ("failed", "closed", "disconnected"):
                self._pcs.discard(pc)

        @pc.on("datachannel")
        def _on_datachannel(channel: Any) -> None:
            self._channels.append(channel)

            @channel.on("message")
            def _on_message(message: str) -> None:
                async def _respond() -> None:
                    try:
                        cmd = json.loads(message)
                        resp = await self._on_command(cmd)
                        channel.send(json.dumps(resp))
                    except Exception as exc:
                        channel.send(json.dumps({"error": str(exc)}))

                asyncio.create_task(_respond())

        pc.addTrack(self._ensure_track())

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=sdp, type=type_)
        )
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        # Brief pause to gather initial ICE candidates
        await asyncio.sleep(0.3)

        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    async def close(self) -> None:
        for pc in list(self._pcs):
            await pc.close()
        self._pcs.clear()


# ── Isaac Sim Bridge ──────────────────────────────────────────────

class IsaacSimBridge:
    """Manages Isaac Sim world, URDF loading, and remote command serving.

    This class is intended to be instantiated inside an Isaac Sim process.
    """

    def __init__(self, websocket_port: int = 8765, mjpeg_port: int = 8766) -> None:
        if not HAS_ISAAC:
            raise RuntimeError(
                "Isaac Sim APIs are not available. "
                "Run this script inside Isaac Sim's Python environment."
            )
        if not HAS_WEBSOCKETS:
            raise RuntimeError("websockets package is required. pip install websockets")

        self._ws_port = websocket_port
        self._mjpeg_port = mjpeg_port
        self._world: World | None = None
        self._robot: Any = None
        self._joint_names: list[str] = []
        self._playing = False
        self._stage_open = False
        self._mjpeg_clients: set[asyncio.StreamWriter] = set()
        self._frame_task: asyncio.Task | None = None
        self._server: Any = None
        self._mjpeg_server: asyncio.AbstractServer | None = None
        self._webrtc: WebRTCManager | None = None

        if HAS_AIORRTC:
            self._webrtc = WebRTCManager(
                frame_source=self._capture_frame_rgb,
                on_command=self._handle_command,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def startup(self) -> None:
        """Create world and start servers."""
        self._world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
        self._world.scene.add_default_ground_plane()
        self._stage_open = True
        logger.info("IsaacSimBridge started")

    async def run(self) -> None:
        """Block and serve WebSocket + MJPEG forever."""
        loop = asyncio.get_running_loop()
        self._server = await websockets.serve(
            self._handle_client, "0.0.0.0", self._ws_port
        )
        self._mjpeg_server = await asyncio.start_server(
            self._handle_mjpeg_client, "0.0.0.0", self._mjpeg_port
        )
        logger.info(
            "Servers listening",
            extra={
                "websocket": self._ws_port,
                "mjpeg": self._mjpeg_port,
                "webrtc": HAS_AIORRTC,
            },
        )
        await asyncio.Future()  # run forever

    def shutdown(self) -> None:
        if self._server:
            self._server.close()
        if self._mjpeg_server:
            self._mjpeg_server.close()
        if self._frame_task:
            self._frame_task.cancel()
        if self._webrtc:
            asyncio.create_task(self._webrtc.close())
        logger.info("IsaacSimBridge shut down")

    # ------------------------------------------------------------------
    # WebSocket command handling
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
            return self._cmd_play()
        if action == "pause":
            return self._cmd_pause()
        if action == "step":
            steps = cmd.get("steps", 1)
            return self._cmd_step(steps)
        if action == "reset":
            return self._cmd_reset()
        if action == "get_state":
            return self._cmd_get_state()
        if action == "set_joints":
            return self._cmd_set_joints(cmd.get("positions", {}))
        if action == "webrtc_offer":
            if not self._webrtc:
                return {"error": "WebRTC not available (aiortc missing)"}
            answer = await self._webrtc.handle_offer(
                cmd.get("sdp", ""), cmd.get("type", "offer")
            )
            return {"ok": True, "webrtc_answer": answer}

        return {"error": f"unknown action: {action}"}

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def _cmd_load_urdf(self, path: str) -> dict[str, Any]:
        if not self._world:
            return {"error": "world not initialized"}

        urdf_path = Path(path)
        if not urdf_path.exists():
            return {"error": f"URDF not found: {path}"}

        # Import URDF into current stage via Isaac Sim URDF importer
        import_config = _urdf.ImportConfig()
        import_config.fix_base = False
        import_config.replace_cylinder_with_capsule = True
        import_config.merge_fixed_joints = False
        import_config.self_collision = False

        result, robot_path = _urdf.UrdfFile().import_robot(
            str(urdf_path),
            str(urdf_path.parent),
            "",
            import_config,
        )

        if not result:
            return {"error": "URDF import failed"}

        # Attempt to get robot articulation
        from omni.isaac.core.articulations import Articulation

        self._robot = Articulation(robot_path)
        self._robot.initialize()
        self._joint_names = self._robot.dof_names

        return {
            "ok": True,
            "joints": self._joint_names,
            "dof_count": len(self._joint_names),
        }

    def _cmd_play(self) -> dict[str, Any]:
        if self._world:
            self._world.play()
        self._playing = True
        return {"ok": True, "playing": True}

    def _cmd_pause(self) -> dict[str, Any]:
        if self._world:
            self._world.pause()
        self._playing = False
        return {"ok": True, "playing": False}

    def _cmd_step(self, steps: int = 1) -> dict[str, Any]:
        if not self._world:
            return {"error": "world not initialized"}
        for _ in range(steps):
            self._world.step(render=True)
        return {"ok": True, "state": self._get_state_dict()}

    def _cmd_reset(self) -> dict[str, Any]:
        if self._world:
            self._world.reset()
        if self._robot:
            self._robot.initialize()
        return {"ok": True}

    def _cmd_get_state(self) -> dict[str, Any]:
        return {"ok": True, "state": self._get_state_dict()}

    def _cmd_set_joints(self, positions: dict[str, float]) -> dict[str, Any]:
        if not self._robot:
            return {"error": "no robot loaded"}
        for name, pos in positions.items():
            idx = self._robot.get_dof_index(name)
            if idx >= 0:
                self._robot.set_joint_position(name, pos)
        return {"ok": True}

    # ------------------------------------------------------------------
    # State extraction
    # ------------------------------------------------------------------

    def _get_state_dict(self) -> dict[str, Any]:
        joints = []
        if self._robot:
            positions = self._robot.get_joint_positions()
            velocities = self._robot.get_joint_velocities()
            for i, name in enumerate(self._joint_names):
                joints.append(
                    {
                        "name": name,
                        "position": float(positions[i]) if i < len(positions) else 0.0,
                        "velocity": float(velocities[i]) if i < len(velocities) else 0.0,
                    }
                )

        base_pose = None
        if self._robot and hasattr(self._robot, "get_root_pose"):
            pose = self._robot.get_root_pose()
            base_pose = {
                "position": [float(pose.p.x), float(pose.p.y), float(pose.p.z)],
                "orientation": [
                    float(pose.r.x),
                    float(pose.r.y),
                    float(pose.r.z),
                    float(pose.r.w),
                ],
            }

        return {
            "joints": joints,
            "base_pose": base_pose,
            "playing": self._playing,
            "timestamp": self._world.current_time if self._world else 0.0,
        }

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def _capture_frame_rgb(self) -> "np.ndarray | None":
        """Capture the active viewport and return an RGB numpy array [H, W, 3]."""
        if not HAS_PIL or not HAS_ISAAC:
            return None
        try:
            vp = get_active_viewport()
            texture = vp.get_texture()
            if texture is None:
                return None

            import omni.hydra

            device = omni.hydra.get_device()
            arr = device.read_texture(texture)
            if arr is None:
                return None

            # arr is typically [H, W, 4] RGBA — return RGB slice
            return arr[:, :, :3]
        except Exception as exc:
            logger.warning("Frame capture failed", extra={"error": str(exc)})
            return None

    def _capture_frame_jpeg(self) -> bytes | None:
        """Capture the active viewport and return JPEG bytes."""
        arr = self._capture_frame_rgb()
        if arr is None:
            return None
        img = Image.fromarray(arr, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # MJPEG streaming
    # ------------------------------------------------------------------

    async def _handle_mjpeg_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Serve an MJPEG stream over HTTP."""
        self._mjpeg_clients.add(writer)
        logger.info("MJPEG client connected")

        # Send HTTP headers
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
                frame = self._capture_frame_jpeg()
                if frame:
                    boundary = b"--frame\r\n"
                    ct = b"Content-Type: image/jpeg\r\n"
                    cl = f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    writer.write(boundary + ct + cl + frame + b"\r\n")
                    await writer.drain()
                await asyncio.sleep(1.0 / 30.0)  # 30 FPS
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            self._mjpeg_clients.discard(writer)
            writer.close()
            logger.info("MJPEG client disconnected")


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Isonome Isaac Sim Bridge")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket command port")
    parser.add_argument("--mjpeg-port", type=int, default=8766, help="MJPEG stream port")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    bridge = IsaacSimBridge(
        websocket_port=args.ws_port,
        mjpeg_port=args.mjpeg_port,
    )
    bridge.startup()
    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
