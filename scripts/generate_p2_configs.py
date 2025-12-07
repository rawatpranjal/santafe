#!/usr/bin/env python3
"""Generate missing P2 tournament configs for Ringuette, BGAN, Staecker, Ledyard."""

from pathlib import Path

# Base directory
CONF_DIR = Path(__file__).parent.parent / "conf" / "experiment" / "p2_tournament"

# Environment configs (matching existing P2 configs)
ENVS = {
    "base": {
        "gametype": 6453,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 3,
        "num_steps": 75,
        "max_price": 2000,
    },
    "bbbs": {
        "gametype": 6453,
        "num_buyers": 6,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_periods": 3,
        "num_steps": 50,
        "max_price": 2000,
    },
    "bsss": {
        "gametype": 6453,
        "num_buyers": 2,
        "num_sellers": 6,
        "num_tokens": 4,
        "num_periods": 3,
        "num_steps": 50,
        "max_price": 2000,
    },
    "eql": {
        "gametype": 0,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 3,
        "num_steps": 75,
        "max_price": 2000,
    },
    "lad": {
        "gametype": 0,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 3,
        "num_steps": 75,
        "max_price": 2000,
    },
    "per": {
        "gametype": 6453,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 1,
        "num_steps": 75,
        "max_price": 2000,
    },
    "ran": {
        "gametype": 7,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 3,
        "num_steps": 50,
        "max_price": 3000,
    },
    "shrt": {
        "gametype": 6453,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 3,
        "num_steps": 25,
        "max_price": 2000,
    },
    "sml": {
        "gametype": 6453,
        "num_buyers": 2,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_periods": 3,
        "num_steps": 50,
        "max_price": 2000,
    },
    "tok": {
        "gametype": 6453,
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 1,
        "num_periods": 3,
        "num_steps": 25,
        "max_price": 2000,
    },
}

# New strategies to add (short code -> class name)
NEW_STRATEGIES = {
    "ring": "Ringuette",
    "bgan": "BGAN",
    "staecker": "Staecker",
    "el": "Ledyard",
}


def generate_ctrl_config(
    strategy_code: str, strategy_name: str, env_code: str, env_cfg: dict
) -> str:
    """Generate control config (1 challenger vs 7 ZIC)."""
    nb = env_cfg["num_buyers"]
    ns = env_cfg["num_sellers"]

    # One challenger buyer, rest ZIC
    buyers = [strategy_name] + ["ZIC"] * (nb - 1)
    sellers = ["ZIC"] * ns

    return f"""# @package _global_
# Part 2: {strategy_name} vs {nb + ns - 1} ZIC, {env_code.upper()} environment

experiment:
  name: "p2_ctrl_{strategy_code}_{env_code}"
  num_rounds: 2

market:
  min_price: 1
  max_price: {env_cfg['max_price']}
  num_tokens: {env_cfg['num_tokens']}
  num_periods: {env_cfg['num_periods']}
  num_steps: {env_cfg['num_steps']}
  gametype: {env_cfg['gametype']}

agents:
  buyer_types: {buyers}
  seller_types: {sellers}
"""


def generate_self_config(
    strategy_code: str, strategy_name: str, env_code: str, env_cfg: dict
) -> str:
    """Generate self-play config (8 identical agents)."""
    nb = env_cfg["num_buyers"]
    ns = env_cfg["num_sellers"]

    buyers = [strategy_name] * nb
    sellers = [strategy_name] * ns

    return f"""# @package _global_
# Part 2: {strategy_name} self-play, {env_code.upper()} environment

experiment:
  name: "p2_self_{strategy_code}_{env_code}"
  num_rounds: 2

market:
  min_price: 1
  max_price: {env_cfg['max_price']}
  num_tokens: {env_cfg['num_tokens']}
  num_periods: {env_cfg['num_periods']}
  num_steps: {env_cfg['num_steps']}
  gametype: {env_cfg['gametype']}

agents:
  buyer_types: {buyers}
  seller_types: {sellers}
"""


def main():
    ctrl_dir = CONF_DIR / "ctrl"
    self_dir = CONF_DIR / "self"

    ctrl_dir.mkdir(parents=True, exist_ok=True)
    self_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for strategy_code, strategy_name in NEW_STRATEGIES.items():
        for env_code, env_cfg in ENVS.items():
            # Control config
            ctrl_path = ctrl_dir / f"p2_ctrl_{strategy_code}_{env_code}.yaml"
            ctrl_content = generate_ctrl_config(strategy_code, strategy_name, env_code, env_cfg)
            ctrl_path.write_text(ctrl_content)
            print(f"Created: {ctrl_path.name}")
            count += 1

            # Self-play config
            self_path = self_dir / f"p2_self_{strategy_code}_{env_code}.yaml"
            self_content = generate_self_config(strategy_code, strategy_name, env_code, env_cfg)
            self_path.write_text(self_content)
            print(f"Created: {self_path.name}")
            count += 1

    print(f"\nTotal configs created: {count}")


if __name__ == "__main__":
    main()
