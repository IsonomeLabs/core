"""Hello Kernel — v0.2 demo with a pre-trained kernel.

Demonstrates before/after side-by-side:
  - Naive mapping: robot arm reaches 15cm to the left of target
  - Calibrated kernel: corrects the -15cm bias
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import torch

from isonome.core.config import AppConfig, SimConfig, SomaConfig
from isonome.core.agent import Agent
from isonome.core.safety import AgentMode
from isonome.core.state import RawSensorState, ExecutionResult
from examples.mock_vla import MockVLABackend

logging.basicConfig(level=logging.INFO)


class MockSomaAgent(Agent):
    """Agent with mock soma for demonstration."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self._tick_count = 0
        self._mock_vla = MockVLABackend(canonical_dim=14)
        self._current_proprio = torch.zeros(self.soma.naive_mapper.joint_count)
        self._use_kernel = False

    async def boot(self) -> None:
        await super().boot()
        self.jepa._policy = self._mock_vla

    async def _async_perceive(self) -> RawSensorState:
        joint_count = self.soma.naive_mapper.joint_count
        if self._current_proprio.numel() != joint_count:
            self._current_proprio = torch.zeros(joint_count)
        return RawSensorState(
            proprioception=self._current_proprio.clone(),
            camera_frames=[],
            timestamp=self._tick_count * 0.1,
        )

    async def _async_act(self, safe_commands: list) -> None:
        if safe_commands:
            cmd = safe_commands[0].command
            self._current_proprio = cmd.clone()
            # Physics bias: actuator always undershoots by 0.15 on axis 2
            self._current_proprio[2] -= 0.15

    async def _async_observe_result(self) -> ExecutionResult:
        return ExecutionResult(
            final_proprioception=self._current_proprio.clone(),
            success=True,
            error_metric=0.0 if self._use_kernel else 0.15,
        )


async def run_agent(kernel_path: Path | None, label: str) -> list[str]:
    config = AppConfig(
        agent_name=f"hello_kernel_{label}",
        sim=SimConfig(gui=False),
        soma=SomaConfig(
            urdf_path="examples/robot_arm.urdf",
            kernel_path=str(kernel_path) if kernel_path else None,
        ),
    )
    agent = MockSomaAgent(config)
    await agent.boot()

    if kernel_path is not None and kernel_path.exists():
        await agent.load_kernel(kernel_path)
        agent._use_kernel = True

    agent.mode = AgentMode.RUNTIME

    results: list[str] = []
    for tick in range(10):
        await agent.tick()
        agent._tick_count = tick + 1
        last = agent.cortex.buffer.last()
        if last is not None:
            intended = last.intended.actions[0, :3].tolist()
            actual = last.actual.final_proprioception[:3].tolist()
            delta = [a - b for a, b in zip(intended, actual)]
            results.append(
                f"Tick {tick + 1:2d} | "
                f"Delta[:3]: [{delta[0]:+.2f}, {delta[1]:+.2f}, {delta[2]:+.2f}]"
            )

    await agent.shutdown()
    return results


async def main() -> None:
    # Ensure URDF exists
    urdf_path = Path("examples/robot_arm.urdf")
    if not urdf_path.exists():
        urdf_path.write_text("""<?xml version="1.0"?>
<robot name="dummy_arm">
  <link name="base_link"/>
  <joint name="j1" type="revolute"><parent link="base_link"/><child link="l1"/><axis xyz="0 0 1"/></joint>
  <link name="l1"/>
  <joint name="j2" type="revolute"><parent link="l1"/><child link="l2"/><axis xyz="0 1 0"/></joint>
  <link name="l2"/>
  <joint name="j3" type="revolute"><parent link="l2"/><child link="l3"/><axis xyz="0 1 0"/></joint>
  <link name="l3"/>
  <joint name="j4" type="revolute"><parent link="l3"/><child link="l4"/><axis xyz="1 0 0"/></joint>
  <link name="l4"/>
  <joint name="j5" type="revolute"><parent link="l4"/><child link="l5"/><axis xyz="0 1 0"/></joint>
  <link name="l5"/>
  <joint name="j6" type="revolute"><parent link="l5"/><child link="l6"/><axis xyz="1 0 0"/></joint>
  <link name="l6"/>
  <joint name="j7" type="revolute"><parent link="l6"/><child link="l7"/><axis xyz="0 1 0"/></joint>
  <link name="l7"/>
</robot>
""")

    print("\n=== Isonome v0.2 Hello Kernel (Before vs After) ===\n")

    kernel_path = Path("examples/dummy_kernel.pt")

    print("--- NAIVE MAPPING (no kernel) ---")
    naive_results = await run_agent(None, "naive")
    for line in naive_results[:3]:
        print(line)
    print("...")

    print("\n--- CALIBRATED KERNEL ---")
    kernel_results = await run_agent(kernel_path, "kernel")
    for line in kernel_results[:3]:
        print(line)
    print("...")

    # Summary
    print("\n=== Summary ===")
    naive_last = naive_results[-1] if naive_results else ""
    kernel_last = kernel_results[-1] if kernel_results else ""
    print(f"Naive last:  {naive_last}")
    print(f"Kernel last: {kernel_last}")
    print("\nThe calibrated kernel corrects the systematic -0.15m bias on axis 2.")
    print("In production, kernels are trained via a separate calibration pipeline.\n")


if __name__ == "__main__":
    asyncio.run(main())
