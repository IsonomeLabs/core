# isonome-framework Recovery Guide

This file documents everything needed to rebuild the agent environment
from scratch after a container restart, computer reboot, or fresh instance.

## Quick Start (run these commands in order)

```bash
# 1. Clone repos
git clone git@github.com:IsonomeLabs/core.git /root/isonome-framework
git clone git@github.com:IsonomeLabs/research.git /root/Developer/TOPIC_RESEARCH

# 2. Set git identity
cd /root/isonome-framework && git config user.name "jverene" && git config user.email "jverene@users.noreply.github.com"
cd /root/Developer/TOPIC_RESEARCH && git config user.name "jverene" && git config user.email "jverene@users.noreply.github.com"

# 3. Install deps
cd /root/isonome-framework && pip install pydantic numpy scipy networkx pytest pytest-asyncio pytest-cov

# 4. Create required directories
mkdir -p /root/isonome-framework/dashboard
mkdir -p /root/isonome-framework/research
mkdir -p /root/Developer/TOPIC_RESEARCH/{qml,multi-agent,robotics,infrastructure,social,content,policy,network}

# 5. Verify
cd /root/isonome-framework && python -m pytest tests/ -q
```

## Cron Job Setup

The single cron job rotates through 3 roles every 15 minutes.
After restoring, recreate it via:

```bash
hermes cron create "every 15m" --prompt "[see cron-prompt.txt in this repo]"
```

Or use the Hermes CLI to set it up. The job handles:
- **Role A (2/3 runs)**: Framework improvements
- **Role B (1/6 runs)**: Frontend dashboard
- **Role C (1/6 runs)**: Research (framework + TOPIC_RESEARCH)

## SSH Keys Needed

These must be added to GitHub deploy keys BEFORE cloning:

### Core repo (IsonomeLabs/core)
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJhTlTHheTrDdpoamyxVkOIrbBfVMCnXzbRw5xCY2Tz+ isonome-cron@docker
```

### Research repo (IsonomeLabs/research)
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIPV3k5U6rwKso271kSh6VIPaiBb7sA4MTzorO5cPawvw isonome-research@docker
```

## Git Remotes

```
IsonomeLabs/core:    git@github.com:IsonomeLabs/core.git
IsonomeLabs/research: git@github.com-research:IsonomeLabs/research.git
```

SSH config for research repo:
```
Host github.com-research
    HostName github.com
    IdentityFile ~/.ssh/id_ed25519_research
    User git
```

## Infrastructure Files

| File | Location | Purpose |
|------|----------|---------|
| knowledge-bank.md | IsonomeLabs/research | Cross-agent intelligence, blockers, patterns |
| dependency-tracker.md | IsonomeLabs/research | Blocker → research → unblock workflow |
| research-directions.md | IsonomeLabs/research | 33 rated research topics |
| research-directions.md | IsonomeLabs/core | 6 framework-specific research topics |
| .cron-role-state | /root/isonome-framework/ | Tracks which role runs next |

## Known Issues

- Hermes cron scheduler only processes one job at a time
- New cron jobs created via tools don't fire if an older job exists
- Workaround: consolidate everything into one rotating job
