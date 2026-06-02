from __future__ import annotations
import asyncio
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(name="isonome", help="Isonome -- Agentic robotics framework")


INIT_FILES: dict[str, str] = {
    "main.py": '''"""Robot entrypoint -- Isonome bootstraps Agent lifecycle."""
import asyncio
from isonome.core.app import IsonomeApp
from isonome.core.config import AppConfig


async def main():
    app = IsonomeApp(AppConfig())
    await app.run(duration_s=60)


if __name__ == "__main__":
    asyncio.run(main())
''',
    "config.yaml": '''reflex:
  frequency_hz: 100.0
  control_freq_hz: 100.0
  policy_freq_hz: 1.0
jepa:
  frequency_hz: 1.0
  backend: openvla
cortex:
  frequency_hz: 0.5
soma:
  urdf_path: urdf/robot.urdf
safety:
  permit_boot_adaptation: false
  error_repeat_threshold: 3
sim:
  engine: pybullet
  gui: false
''',
    "layers/__init__.py": "",
    "layers/reflex.py": '''from isonome.core.layers.reflex import ReflexLayer


class MyReflex(ReflexLayer):
    pass
''',
    "layers/jepa.py": '''from isonome.core.layers.jepa import JEPALayer


class MyJEPA(JEPALayer):
    pass
''',
    "layers/cortex.py": '''from isonome.core.layers.cortex import CortexLayer


class MyCortex(CortexLayer):
    pass
''',
    "layers/plasticity.py": '''from isonome.core.layers.plasticity import PlasticityLayer


class MyPlasticity(PlasticityLayer):
    pass
''',
    "sim/world.json": '''{
  "ground": true,
  "obstacles": [],
  "robots": []
}''',
    "tests/__init__.py": "",
    "tests/test_agent.py": '''import pytest
from isonome.core.config import AppConfig
from isonome.core.agent import Agent
from isonome.core.safety import AgentMode


@pytest.mark.asyncio
async def test_agent_boot_shutdown():
    agent = Agent(AppConfig())
    await agent.boot()
    assert agent.mode == AgentMode.IDLE
    await agent.shutdown()
    assert agent.mode == AgentMode.IDLE
'''
}


@app.command()
def init(name: str) -> None:
    """Scaffold a new robot project."""
    base = Path(name)
    if base.exists():
        typer.echo(f"Error: {name}/ already exists", err=True)
        raise typer.Exit(1)

    base.mkdir(parents=True)

    # isonome.toml manifest
    (base / "isonome.toml").write_text(f'[tool.isonome]\nagent_name = "{name}"\n')

    for rel_path, content in INIT_FILES.items():
        file_path = base / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    typer.echo(f"Scaffolded {name}/ with default layers and config")


@app.command()
def sim() -> None:
    """Run robot in simulation mode."""
    typer.echo("Starting simulation... (stub)")
    # TODO: load config, create SimBridge, run Agent


@app.command()
def run() -> None:
    """Run robot on hardware."""
    typer.echo("Starting on hardware... (stub)")
    # TODO: load config, create HardwareBridge, run Agent


@app.command()
def deploy() -> None:
    """Push to physical robot (stub)."""
    typer.echo("Deploying... (stub)")


if __name__ == "__main__":
    app()
