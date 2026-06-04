from __future__ import annotations

import asyncio
import signal
from pathlib import Path
from isonome.core.config import AppConfig
from isonome.core.agent import Agent
from isonome.utils.logging import setup_logging, get_layer_logger


class IsonomeApp:
    """Application lifecycle manager -- bootstraps Agent from config."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._agent = Agent(config)
        self._logger = get_layer_logger("app")
        self._shutdown_event = asyncio.Event()

    @classmethod
    def from_file(cls, path: Path) -> IsonomeApp:
        if path.suffix in (".yaml", ".yml"):
            config = AppConfig.from_yaml(path)
        elif path.suffix == ".toml":
            config = AppConfig.from_toml(path)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        return cls(config)

    @property
    def agent(self) -> Agent:
        return self._agent

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._handle_signal, sig)

    def _handle_signal(self, sig: signal.Signals) -> None:
        self._logger.info("signal_received", extra={"signal": sig.name})
        self._shutdown_event.set()

    async def run(self, duration_s: float | None = None) -> None:
        setup_logging()
        self._logger.info(
            "app_starting", extra={"agent": self._config.agent_name}
        )
        self._install_signal_handlers()

        async def _run_agent() -> None:
            await self._agent.run(duration_s=duration_s)

        agent_task = asyncio.create_task(_run_agent())
        shutdown_task = asyncio.create_task(self._shutdown_event.wait())

        done, pending = await asyncio.wait(
            [agent_task, shutdown_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        self._logger.info("app_stopped")

    def run_sync(self, duration_s: float | None = None) -> None:
        asyncio.run(self.run(duration_s=duration_s))
