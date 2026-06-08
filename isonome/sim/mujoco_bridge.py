"""MuJoCo bridge for remote URDF simulation.

Provides the same WebSocket + MJPEG + WebRTC interface as IsaacSimBridge
but runs MuJoCo physics natively on macOS / Linux / Windows.

Run:
    python -m isonome.sim.mujoco_bridge --ws-port 8765 --mjpeg-port 8766
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import time
from pathlib import Path
from typing import Any

try:
    import mujoco
    import numpy as np

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False
    np = None  # type: ignore[assignment]

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import websockets

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
    from av import VideoFrame

    HAS_AIORRTC = True
except ImportError:
    HAS_AIORRTC = False


logger = logging.getLogger("isonome.sim.mujoco_bridge")


# ── WebRTC helpers ────────────────────────────────────────────────

if HAS_AIORRTC:
    class _SimVideoTrack(VideoStreamTrack):
        """Push frames from MuJoCo renderer into a WebRTC video track."""

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
else:
    class WebRTCManager:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("aiortc is required: pip install aiortc")


# ── MuJoCo Bridge ─────────────────────────────────────────────────

class MuJoCoBridge:
    """Manages MuJoCo physics, URDF loading, and remote command serving."""

    def __init__(self, websocket_port: int = 8765, mjpeg_port: int = 8766) -> None:
        if not HAS_MUJOCO:
            raise RuntimeError("mujoco is required. pip install mujoco")
        if not HAS_WEBSOCKETS:
            raise RuntimeError("websockets package is required. pip install websockets")

        self._ws_port = websocket_port
        self._mjpeg_port = mjpeg_port
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._joint_names: list[str] = []
        self._joint_map: dict[str, int] = {}  # name -> joint index
        self._playing = False
        self._mjpeg_clients: set[asyncio.StreamWriter] = set()
        self._sim_task: asyncio.Task | None = None
        self._server: Any = None
        self._mjpeg_server: asyncio.AbstractServer | None = None
        self._start_time: float = 0.0
        self._webrtc: WebRTCManager | None = None

        if HAS_AIORRTC:
            self._webrtc = WebRTCManager(
                frame_source=self._capture_frame_rgb,
                on_command=self._handle_command,
            )

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
            "MuJoCoBridge listening",
            extra={
                "websocket": self._ws_port,
                "mjpeg": self._mjpeg_port,
                "webrtc": HAS_AIORRTC,
            },
        )
        self._sim_task = asyncio.create_task(self._sim_loop())
        await asyncio.Future()  # run forever

    def shutdown(self) -> None:
        if self._server:
            self._server.close()
        if self._mjpeg_server:
            self._mjpeg_server.close()
        if self._sim_task:
            self._sim_task.cancel()
        if self._webrtc:
            asyncio.create_task(self._webrtc.close())
        logger.info("MuJoCoBridge shut down")

    # ------------------------------------------------------------------
    # Sim loop
    # ------------------------------------------------------------------

    async def _sim_loop(self) -> None:
        while True:
            if self._playing and self._data is not None and self._model is not None:
                mujoco.mj_step(self._model, self._data)
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
            if self._data is not None and self._model is not None:
                for _ in range(steps):
                    mujoco.mj_step(self._model, self._data)
            return {"ok": True, "state": self._get_state_dict()}
        if action == "reset":
            if self._data is not None and self._model is not None:
                mujoco.mj_resetData(self._model, self._data)
            self._start_time = time.time()
            return {"ok": True}
        if action == "get_state":
            return {"ok": True, "state": self._get_state_dict()}
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
        urdf_path = Path(path)
        if not urdf_path.exists():
            return {"error": f"URDF not found: {path}"}

        try:
            self._model = mujoco.MjModel.from_xml_path(str(urdf_path))
            self._data = mujoco.MjData(self._model)
        except Exception as exc:
            return {"error": f"Failed to load URDF: {exc}"}

        # Extract joint names
        self._joint_names = []
        self._joint_map = {}
        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self._joint_names.append(name)
                self._joint_map[name] = i

        # Initialize renderer
        self._renderer = mujoco.Renderer(self._model, height=480, width=640)

        # Set initial pose
        mujoco.mj_resetData(self._model, self._data)
        self._start_time = time.time()
        self._playing = True

        return {
            "ok": True,
            "joints": self._joint_names,
            "dof_count": len(self._joint_names),
        }

    def _cmd_set_joints(self, positions: dict[str, float]) -> dict[str, Any]:
        if self._data is None or self._model is None:
            return {"error": "no model loaded"}
        for name, pos in positions.items():
            idx = self._joint_map.get(name)
            if idx is None:
                continue
            qposadr = self._model.jnt_qposadr[idx]
            jnt_type = self._model.jnt_type[idx]
            if jnt_type == mujoco.mjtJoint.mjJNT_HINGE:
                self._data.qpos[qposadr] = pos
            elif jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                self._data.qpos[qposadr] = pos
            # Free and ball joints are skipped for simple set
        return {"ok": True}

    def _get_state_dict(self) -> dict[str, Any]:
        joints = []
        if self._data is not None and self._model is not None:
            for name in self._joint_names:
                idx = self._joint_map[name]
                qposadr = self._model.jnt_qposadr[idx]
                qveladr = self._model.jnt_dofadr[idx]
                jnt_type = self._model.jnt_type[idx]

                if jnt_type == mujoco.mjtJoint.mjJNT_HINGE or jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
                    pos = float(self._data.qpos[qposadr])
                    vel = float(self._data.qvel[qveladr])
                elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                    # Ball joint: 4 quat components for pos, 3 angular vel
                    pos = float(self._data.qpos[qposadr])
                    vel = float(self._data.qvel[qveladr])
                else:
                    pos = 0.0
                    vel = 0.0

                joints.append({"name": name, "position": round(pos, 4), "velocity": round(vel, 4)})

        # Get body poses if available
        base_pose = None
        if self._model is not None and self._data is not None and self._model.nbody > 1:
            body_id = 1  # first non-world body
            pos = self._data.xpos[body_id]
            quat = self._data.xquat[body_id]
            base_pose = {
                "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                "orientation": [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])],
            }

        elapsed = time.time() - self._start_time if self._start_time else 0.0

        return {
            "joints": joints,
            "base_pose": base_pose,
            "playing": self._playing,
            "timestamp": elapsed,
        }

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def _capture_frame_rgb(self) -> "np.ndarray | None":
        """Render the current MuJoCo scene and return an RGB numpy array [H, W, 3]."""
        if not HAS_MUJOCO or self._renderer is None or self._data is None:
            return None
        try:
            self._renderer.update_scene(self._data)
            frame = self._renderer.render()
            # frame is [H, W, 3] RGB uint8
            return frame
        except Exception as exc:
            logger.warning("Frame capture failed", extra={"error": str(exc)})
            return None

    def _capture_frame_jpeg(self) -> bytes | None:
        """Render the current MuJoCo scene and return JPEG bytes."""
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
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Isonome MuJoCo Bridge")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket command port")
    parser.add_argument("--mjpeg-port", type=int, default=8766, help="MJPEG stream port")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    bridge = MuJoCoBridge(
        websocket_port=args.ws_port,
        mjpeg_port=args.mjpeg_port,
    )
    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
