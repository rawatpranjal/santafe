#!/usr/bin/env python3
"""Generate missing Santa Fe trader configs for Part 2."""

from pathlib import Path

# Missing Santa Fe traders
MISSING_TRADERS = {
    "gamer": "Gamer",
    "jacobson": "Jacobson",
    "perry": "Perry",
    "lin": "Lin",
    "breton": "Breton",
}

# Environment configs
# env_name: (gametype, max_price, num_steps)
ENV_CONFIGS = {
    "base": (6453, 2000, 75),
    "bbbs": (6453, 2000, 75),
    "bsss": (6453, 2000, 75),
    "eql": (0, 2000, 75),
    "ran": (7, 3000, 50),
    "per": (6453, 2000, 75),
    "shrt": (6453, 2000, 75),
    "tok": (6453, 2000, 75),
    "sml": (6453, 2000, 75),
    "lad": (0, 2000, 75),
}

SELF_TEMPLATE = """# @package _global_
# Part 2: {trader_name} self-play, {env_upper} environment

experiment:
  name: "p2_self_{trader_short}_{env}"
  num_rounds: 2

market:
  min_price: 1
  max_price: {max_price}
  num_tokens: 4
  num_periods: 3
  num_steps: {num_steps}
  gametype: {gametype}

agents:
  buyer_types: ['{trader_name}', '{trader_name}', '{trader_name}', '{trader_name}']
  seller_types: ['{trader_name}', '{trader_name}', '{trader_name}', '{trader_name}']
"""

CTRL_TEMPLATE = """# @package _global_
# Part 2: {trader_name} vs 7 ZIC, {env_upper} environment

experiment:
  name: "p2_ctrl_{trader_short}_{env}"
  num_rounds: 2

market:
  min_price: 1
  max_price: {max_price}
  num_tokens: 4
  num_periods: 3
  num_steps: {num_steps}
  gametype: {gametype}

agents:
  buyer_types: ['{trader_name}', 'ZIC', 'ZIC', 'ZIC']
  seller_types: ['ZIC', 'ZIC', 'ZIC', 'ZIC']
"""


def generate_configs():
    base_dir = Path("conf/experiment/p2_tournament")
    self_dir = base_dir / "self"
    ctrl_dir = base_dir / "ctrl"

    self_count = 0
    ctrl_count = 0

    for trader_short, trader_name in MISSING_TRADERS.items():
        for env, (gametype, max_price, num_steps) in ENV_CONFIGS.items():
            # Self-play config
            self_file = self_dir / f"p2_self_{trader_short}_{env}.yaml"
            self_content = SELF_TEMPLATE.format(
                trader_name=trader_name,
                trader_short=trader_short,
                env=env,
                env_upper=env.upper(),
                gametype=gametype,
                max_price=max_price,
                num_steps=num_steps,
            )
            self_file.write_text(self_content)
            self_count += 1
            print(f"Created: {self_file}")

            # Control config
            ctrl_file = ctrl_dir / f"p2_ctrl_{trader_short}_{env}.yaml"
            ctrl_content = CTRL_TEMPLATE.format(
                trader_name=trader_name,
                trader_short=trader_short,
                env=env,
                env_upper=env.upper(),
                gametype=gametype,
                max_price=max_price,
                num_steps=num_steps,
            )
            ctrl_file.write_text(ctrl_content)
            ctrl_count += 1
            print(f"Created: {ctrl_file}")

    print(f"\nTotal: {self_count} self-play configs, {ctrl_count} ctrl configs")


if __name__ == "__main__":
    generate_configs()
