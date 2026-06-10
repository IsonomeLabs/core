# Iteration 029 — Quick Test Reference

## 1. Unit tests (VLA integration)

```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest tests/test_vla_integration.py -v
```

## 2. Full suite (minus pre-existing async failures)

```bash
PYTHONPATH=$(pwd) .venv/bin/python -m pytest tests/ -v --ignore=tests/dashboard
```

## 3. Main demo — VLA Steering (headless)

```bash
PYTHONPATH=$(pwd) .venv/bin/python examples/vla_steering.py --steps 50
```

## 4. Main demo — with dashboard viewing

Terminal 1 (run the demo with `--serve`):
```bash
PYTHONPATH=$(pwd) .venv/bin/python examples/vla_steering.py --steps 150 --serve
```

Then open http://localhost:8765/sim.html in a browser.

## 5. Custom user command

```bash
PYTHONPATH=$(pwd) .venv/bin/python examples/vla_steering.py --command "reach the red cube" --steps 50
```

## 6. Simpler reach demo (no LLM layer, just VLA → sim)

```bash
PYTHONPATH=$(pwd) .venv/bin/python examples/vla_reach.py --steps 100
```

## 7. Training scaffolding (no-op with mock model)

```bash
PYTHONPATH=$(pwd) .venv/bin/python examples/train_vla.py --model mock --epochs 5
```

## 8. MuJoCo bridge standalone (for manual driving via dashboard)

```bash
PYTHONPATH=$(pwd) .venv/bin/python -m isonome.sim.mujoco_bridge --verbose
```
