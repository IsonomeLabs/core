"""Isaac Lab bridge for GPU-accelerated URDF simulation.

Provides the same WebSocket + MJPEG + WebRTC interface as the other sim
bridges but runs inside an Isaac Lab ``IsaacEnv`` / ``SimulationContext``.
This is the architecture's primary simulation backend.

Run from an environment with ``isaaclab`` installed::

    python -m isonome.sim.isaac_lab_bridge --ws-port 8765 --mjpeg-port 8766
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
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
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

# Isaac Lab imports are guarded — the package is heavy and optional.
try:
    from isaaclab.app import AppLauncher
    from isaaclab.envs import ManagerBasedRLEnv

    HAS_ISAACLAB = True
except ImportError:
    HAS_ISAACLAB = False
    AppLauncher = None  # type: ignore[misc, assignment]
    ManagerBasedRLEnv = None  # type: ignore[misc, assignment]


logger = logging.getLogger("isonome.sim.isaac_lab_bridge")


# ── WebRTC helpers ────────────────────────────────────────────────

if HAS_AIORRTC:

    class _SimVideoTrack(VideoStreamTrack):
        """Push frames from Isaac Lab viewport into a WebRTC video track."""

        kind = "video"

        def __init__(self, frame_source: Any) -> None:
            super().__init__()
            self._frame_source = frame_source

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


# ── Isaac Lab Bridge ──────────────────────────────────────────────

class IsaacLabBridge:
    """Manages an Isaac Lab environment, URDF loading, and remote commands.

    This class is intended to be instantiated inside a Python environment where
    ``isaaclab`` is installed.  It uses ``ManagerBasedRLEnv`` when a task config
    is provided, otherwise it boots a minimal ``SimulationContext`` and loads
    the URDF as an ``Articulation``.
    """

    def __init__(
        self,
        websocket_port: int = 8765,
        mjpeg_port: int = 8766,
        headless: bool = True,
        device: str = "cuda:0",
    ) -> None:
        if not HAS_ISAACLAB:
            raise RuntimeError(
                "Isaac Lab is required. "
                "Install with the isaaclab extras or activate the Isaac Lab conda env."
            )
        if not HAS_WEBSOCKETS:
            raise RuntimeError("websockets package is required. pip install websockets")

        self._ws_port = websocket_port
        self._mjpeg_port = mjpeg_port
        self._headless = headless
        self._device = device

        self._app_launcher: Any = None
        self._env: ManagerBasedRLEnv | None = None
        self._articulation: Any = None
        self._sim: Any = None
        self._joint_names: list[str] = []
        self._joint_map: dict[str, int] = {}
        self._playing = False
        self._mjpeg_clients: set[asyncio.StreamWriter] = set()
        self._sim_task: asyncio.Task | None = None
        self._server: Any = None
        self._mjpeg_server: asyncio.AbstractServer | None = None
        self._start_time: float = 0.0
        self._webrtc: WebRTCManager | None = None
        self._current_intent: str = ""
        self._last_action: np.ndarray | None = None
        self._action_lpf_alpha: float = 0.3

        if HAS_AIORRTC:
            self._webrtc = WebRTCManager(
                frame_source=self._capture_frame_rgb,
                on_command=self._handle_command,
            )

    @property
    def joint_count(self) -> int:
        return len(self._joint_names)

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
            "IsaacLabBridge listening",
            extra={
                "websocket": self._ws_port,
                "mjpeg": self._mjpeg_port,
                "webrtc": HAS_AIORRTC,
                "headless": self._headless,
                "device": self._device,
            },
        )
        self._sim_task = asyncio.create_task(
            self._sim_loop(), name="isaac_lab_physics_loop"
        )
        asyncio.create_task(self._watch_task(self._sim_task, "isaac_lab_physics_loop"))

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            logger.info("IsaacLabBridge main future cancelled")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if getattr(self, "_shutting_down", False):
            return
        self._shutting_down = True
        logger.info("IsaacLabBridge shutting down")
        if self._server:
            self._server.close()
        if self._mjpeg_server:
            self._mjpeg_server.close()
        if self._sim_task:
            self._sim_task.cancel()
        if self._env is not None:
            try:
                self._env.close()
            except Exception as exc:
                logger.warning("Error closing Isaac Lab env: %s", exc)
        if self._webrtc:
            asyncio.create_task(self._webrtc.close())
        logger.info("IsaacLabBridge shut down")

    async def _watch_task(self, task: asyncio.Task, name: str) -> None:
        """Log exceptions from background tasks so they don't vanish silently."""
        try:
            await task
        except asyncio.CancelledError:
            logger.debug("Task %s cancelled", name)
        except Exception as exc:
            logger.exception("Task %s crashed: %s", name, exc)
            self.shutdown()

    # ------------------------------------------------------------------
    # Sim loop
    # ------------------------------------------------------------------

    async def _sim_loop(self) -> None:
        """Step Isaac Lab physics at 60 Hz while ``play`` is active."""
        while True:
            if self._playing and self._env is not None:
                try:
                    self._env.step(None)
                except Exception as exc:
                    logger.exception("Isaac Lab step failed: %s", exc)
                    self._playing = False
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
            return self._cmd_step(cmd.get("steps", 1))
        if action == "reset":
            return self._cmd_reset()
        if action == "get_state":
            return {"ok": True, "state": self._get_state_dict()}
        if action == "set_joints":
            return self._cmd_set_joints(cmd.get("positions", {}))
        if action == "set_intent":
            self._current_intent = cmd.get("text", "")
            return {"ok": True, "intent": self._current_intent}
        if action == "get_intent":
            return {"ok": True, "intent": self._current_intent}
        if action == "get_observation":
            obs = self.get_observation(self._current_intent)
            obs_json: dict[str, Any] = {
                "intent": obs.get("intent", ""),
                "timestamp": obs.get("timestamp", 0.0),
            }
            proprio = obs.get("proprioception")
            if hasattr(proprio, "tolist"):
                obs_json["proprioception"] = proprio.tolist()
            else:
                obs_json["proprioception"] = proprio
            images = obs.get("image")
            if images is None:
                obs_json["n_cameras"] = 0
            elif isinstance(images, list):
                obs_json["n_cameras"] = len(images)
            else:
                obs_json["n_cameras"] = 1
            obs_json["image"] = None
            return {"ok": True, "observation": obs_json}
        if action == "apply_action":
            return self._cmd_apply_action(cmd.get("action", []))
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
            # Launch Isaac Lab/Isaac Sim app if not already running
            if self._app_launcher is None:
                self._app_launcher = AppLauncher(
                    headless=self._headless,
                    offscreen_render=self._headless,
                )
                self._app_launcher.app_window.update()

            # Build a minimal ManagerBasedRLEnv from a default CartPole-style
            # task config and then replace the robot asset with the requested
            # URDF.  This keeps the bridge dependency-free in terms of custom
            # task files while still exercising Isaac Lab's RL env pipeline.
            #
            # Users who already have an Isaac Lab task can pass
            # ``task_cfg_path`` via engine_options to bypass this default.
            from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
            from isaaclab.envs import ManagerBasedRLEnvCfg
            from isaaclab.managers import ObservationManagerCfg, RewardManagerCfg
            from isaaclab.scene import InteractiveSceneCfg
            from isaaclab.assets import ArticulationCfg

            asset_cfg = ArticulationCfg(
                prim_path="/World/Robot",
                spawn=urdf_path,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 0.5),
                    joint_pos={},
                ),
                actuators={},
            )
            scene_cfg = InteractiveSceneCfg(
                robot=asset_cfg,
                env_spacing=2.0,
            )
            env_cfg = ManagerBasedRLEnvCfg(
                scene=scene_cfg,
                observations=ObservationManagerCfg(),
                rewards=RewardManagerCfg(),
            )
            self._env = ManagerBasedRLEnv(cfg=env_cfg)
            self._sim = self._env.sim
            self._articulation = self._env.scene["robot"]

            # Cache joint names
            self._joint_names = list(self._articulation.joint_names)
            self._joint_map = {name: i for i, name in enumerate(self._joint_names)}

            self._start_time = time.time()
            self._playing = True

            return {
                "ok": True,
                "joints": self._joint_names,
                "dof_count": len(self._joint_names),
            }
        except Exception as exc:
            logger.exception("Isaac Lab URDF load failed")
            return {"error": f"Failed to load URDF in Isaac Lab: {exc}"}

    def _cmd_step(self, steps: int = 1) -> dict[str, Any]:
        if self._env is None:
            return {"error": "no model loaded"}
        for _ in range(steps):
            self._env.step(None)
        return {"ok": True, "state": self._get_state_dict()}

    def _cmd_reset(self) -> dict[str, Any]:
        if self._env is not None:
            self._env.reset()
        self._start_time = time.time()
        return {"ok": True}

    def _cmd_set_joints(self, positions: dict[str, float]) -> dict[str, Any]:
        if self._articulation is None:
            return {"error": "no robot loaded"}
        joint_pos = self._articulation.data.default_joint_pos.clone()
        for name, pos in positions.items():
            idx = self._joint_map.get(name)
            if idx is not None:
                joint_pos[:, idx] = float(pos)
        self._articulation.write_joint_state_to_sim(
            joint_pos, self._articulation.data.default_joint_vel
        )
        self._articulation.set_joint_position_target(joint_pos)
        return {"ok": True}

    def get_observation(self, intent: str = "") -> dict[str, Any]:
        """Package current sim state into a VLA observation."""
        return {
            "image": self._capture_frame_rgb(),
            "proprioception": self._get_proprio(),
            "intent": intent,
            "timestamp": time.time(),
        }

    def _get_proprio(self) -> "np.ndarray":
        """Return joint positions and velocities from the articulation."""
        if self._articulation is None:
            return np.zeros(0, dtype=np.float32)
        pos = self._articulation.data.joint_pos.cpu().numpy().flatten()
        vel = self._articulation.data.joint_vel.cpu().numpy().flatten()
        return np.concatenate([pos, vel]).astype(np.float32)

    def _cmd_apply_action(self, action: list[float]) -> dict[str, Any]:
        """Apply a VLA action (delta positions) with optional smoothing."""
        if self._articulation is None:
            return {"error": "no robot loaded"}
        action_arr = np.asarray(action, dtype=np.float32)
        if self._last_action is not None and action_arr.shape == self._last_action.shape:
            action_arr = (
                self._action_lpf_alpha * action_arr
                + (1.0 - self._action_lpf_alpha) * self._last_action
            )
        self._last_action = action_arr.copy()

        current_pos = self._articulation.data.joint_pos.cpu().numpy().flatten()
        target = current_pos.copy()
        for i in range(min(len(action_arr), len(target))):
            target[i] += float(action_arr[i])
        target_tensor = (
            self._articulation.data.default_joint_pos.clone().flatten()
        )
        target_tensor[: len(target)] = (
            self._to_tensor(target, device=self._articulation.data.joint_pos.device)
        )
        self._articulation.set_joint_position_target(target_tensor)
        return {"ok": True}

    def _to_tensor(self, arr: "np.ndarray", device: Any) -> Any:
        """Convert a numpy array to the Isaac Lab backend tensor type."""
        import torch

        return torch.from_numpy(arr).to(device=device, dtype=torch.float32)

    def _get_state_dict(self) -> dict[str, Any]:
        joints = []
        if self._articulation is not None:
            positions = self._articulation.data.joint_pos.cpu().numpy().flatten()
            velocities = self._articulation.data.joint_vel.cpu().numpy().flatten()
            for i, name in enumerate(self._joint_names):
                joints.append(
                    {
                        "name": name,
                        "position": round(float(positions[i]), 4) if i < len(positions) else 0.0,
                        "velocity": round(float(velocities[i]), 4) if i < len(velocities) else 0.0,
                    }
                )

        base_pose = None
        if self._articulation is not None:
            root_pos = self._articulation.data.root_pos_w.cpu().numpy().flatten()
            root_quat = self._articulation.data.root_quat_w.cpu().numpy().flatten()
            if len(root_pos) >= 3 and len(root_quat) >= 4:
                # Isaac Lab uses (w, x, y, z); output (x, y, z, w)
                base_pose = {
                    "position": [float(root_pos[0]), float(root_pos[1]), float(root_pos[2])],
                    "orientation": [
                        float(root_quat[1]),
                        float(root_quat[2]),
                        float(root_quat[3]),
                        float(root_quat[0]),
                    ],
                }

        elapsed = time.time() - self._start_time if self._start_time else 0.0

        return {
            "joints": joints,
            "base_pose": base_pose,
            "playing": self._playing,
            "timestamp": elapsed,
            "intent": self._current_intent,
        }

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def _capture_frame_rgb(self) -> "np.ndarray | None":
        """Capture the current Isaac Lab viewport/render product as RGB."""
        if not HAS_NUMPY:
            return None
        try:
            # Isaac Lab exposes a sensor API; fall back to render-to-numpy.
            if self._env is not None and hasattr(self._env, "render"):
                return np.asarray(self._env.render(mode="rgb_array"))
            return None
        except Exception as exc:
            logger.warning("Frame capture failed", extra={"error": str(exc)})
            return None

    def _capture_frame_jpeg(self) -> bytes | None:
        """Capture the current viewport and return JPEG bytes."""
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
                await asyncio.sleep(1.0 / 30.0)
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
    parser = argparse.ArgumentParser(description="Isonome Isaac Lab Bridge")
    parser.add_argument("--ws-port", type=int, default=8765, help="WebSocket command port")
    parser.add_argument("--mjpeg-port", type=int, default=8766, help="MJPEG stream port")
    parser.add_argument("--urdf", type=str, default="", help="Path to URDF to auto-load on startup")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device for tensors")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    bridge = IsaacLabBridge(
        websocket_port=args.ws_port,
        mjpeg_port=args.mjpeg_port,
        headless=args.headless,
        device=args.device,
    )

    if args.urdf:
        result = bridge._cmd_load_urdf(args.urdf)
        if result.get("ok"):
            logger.info("Auto-loaded %s (%d joints)", args.urdf, result.get("dof_count", 0))
        else:
            logger.error("Failed to auto-load %s: %s", args.urdf, result.get("error"))

    asyncio.run(bridge.run())


if __name__ == "__main__":
    main()
