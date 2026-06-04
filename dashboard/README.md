# Isonome Framework Developer Dashboard

Real-time visualization dashboard for the isonome agent framework.

## Quick Start

```bash
# Run with simulated agent activity (demo mode)
python -m dashboard.server --demo --port 8420

# Run with a live agent (no demo)
python -m dashboard.server --port 8420
```

Then open http://localhost:8420 in your browser.

## Dashboard Panels

### ♉ Tension Axes
Eight horizontal bars showing each tension axis position relative to its default.
- **Cognition axes** (blue): explore↔exploit, shallow↔deep, divergent↔convergent
- **Praxis axes** (orange): autonomy↔safety, sequential↔parallel, verify↔execute
- **Mneme axes** (green): consolidate↔prune, specific↔general
- Oscillating axes glow red with a ⚠ warning

### 📊 Stress Level
Semicircular gauge showing RMS drift across all axes. Color-coded:
- Green (< 0.15): Homeostatic equilibrium
- Yellow (0.15–0.35): Mild perturbation
- Red (> 0.35): High stress — corrective feedback active

### 🏛 Pillar Activity
Per-pillar cards showing axis counts, stress levels, oscillation warnings,
and initialization status for Cognition, Praxis, and Mneme.

### 🧠 Attention Budget
Token utilization bar with active chunk count, entropy estimate,
GC cycles, and pruning statistics.

### 💾 Memory (Mneme)
Three-tier memory display: Working, Episodic, and Semantic counts,
plus consolidation, pruning, rehearsal, and retrieval counters.

### 🎯 Calibration
Confidence calibration metrics: ECE, Bias, MCE, and total predictions.
Task type homeostasis profile count.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard HTML |
| `GET /api/state` | Full agent state as JSON |

## Architecture

- **`server.py`**: Python HTTP server using stdlib. Extracts state from
  live `IsonomeAgent` instances. Demo mode simulates ticks with random feedback.
- **`index.html`**: Single-page dashboard with vanilla JS. Polls `/api/state`
  every second. Dark theme with color-coded pillars.
