# Santa Fe Double Auction v2.0

Replication of the 1993 Santa Fe Double Auction Tournament with modern reinforcement learning agents.

## Project Structure

- `engine/` - Core market engine (AURORA rules)
- `traders/` - Agent implementations (ZIC, Kaplan, ZIP, GD, PPO)
- `envs/` - Gymnasium environment for RL training
- `conf/` - Hydra configuration files
- `tests/` - Test suite

## Quick Start

```bash
# Install dependencies
uv sync --all-extras

# Run tests
pytest

# Train PPO agent
python train_ppo.py experiment=rl/2_1_ppo_vs_zic
```

## Documentation

See `CLAUDE.md` for AI agent protocol and `plan.md` for implementation roadmap.
