#!/usr/bin/env python3
"""Update round-robin configs to use only Santa Fe 1991 traders."""

import os

# Santa Fe 1991 roster (12 traders, no ZIP or GD)
# Split into 6 buyers × 6 sellers with ZIC on buyer side only
BUYERS = ["ZIC", "Skeleton", "Kaplan", "Ringuette", "Gamer", "Perry"]
SELLERS = ["Ledyard", "BGAN", "Staecker", "Jacobson", "Lin", "Breton"]

# Market params by env
ENV_PARAMS = {
    "base": {"gametype": 6453},
    "bbbs": {"gametype": 2},
    "bsss": {"gametype": 1},
    "eql": {"gametype": 6543},
    "ran": {"gametype": 0},
    "per": {"gametype": 6453, "num_periods": 10},
    "shrt": {"gametype": 6453, "num_steps": 35},
    "tok": {"gametype": 6453, "num_tokens": 1},
    "sml": {"gametype": 6453, "min_price": 1, "max_price": 200},
    "lad": {"gametype": 0},
}

CONFIG_DIR = "conf/experiment/p2_tournament/rr"

for env, params in ENV_PARAMS.items():
    config_name = f"p2_rr_mixed_{env}"
    config_path = os.path.join(CONFIG_DIR, f"{config_name}.yaml")

    gametype = params.get("gametype", 6453)
    num_periods = params.get("num_periods", 3)
    num_steps = params.get("num_steps", 75)
    num_tokens = params.get("num_tokens", 4)
    min_price = params.get("min_price", 1)
    max_price = params.get("max_price", 2000)

    config_content = f"""# @package _global_
# Part 2: Round Robin mixed market, {env.upper()} environment
# Santa Fe 1991 traders only (12 traders: 6 buyers x 6 sellers)

experiment:
  name: "{config_name}"
  num_rounds: 2

market:
  min_price: {min_price}
  max_price: {max_price}
  num_tokens: {num_tokens}
  num_periods: {num_periods}
  num_steps: {num_steps}
  gametype: {gametype}

agents:
  buyer_types: {BUYERS}
  seller_types: {SELLERS}
"""

    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"Updated: {config_path}")

print("\nRound-robin configs updated with Santa Fe 1991 roster:")
print(f"  Buyers:  {BUYERS}")
print(f"  Sellers: {SELLERS}")
print(f"  Total unique traders: {len(set(BUYERS + SELLERS))}")
