#!/usr/bin/env python3
"""Generate round-robin configs with ZIP included."""
from pathlib import Path

ENVS = {
    "base": 6453,  # BASE environment
    "bbbs": 9111,  # Big Buyer, Big Seller
    "bsss": 1999,  # Big Seller, Small Seller
    "eql": 5555,  # Equal spreads
    "ran": 0,  # Random (gametype 0)
    "per": 6453,  # Persistent (same as BASE)
    "shrt": 6453,  # Short periods
    "tok": 6453,  # Many tokens
    "sml": 6453,  # Small token values
    "lad": 6453,  # Ladder (same as BASE)
}

# Step counts per environment
STEPS = {
    "base": 75,
    "bbbs": 75,
    "bsss": 75,
    "eql": 75,
    "ran": 50,
    "per": 75,
    "shrt": 75,
    "tok": 75,
    "sml": 75,
    "lad": 75,
}

# Token counts
TOKENS = {
    "base": 4,
    "bbbs": 4,
    "bsss": 4,
    "eql": 4,
    "ran": 4,
    "per": 4,
    "shrt": 4,
    "tok": 8,
    "sml": 4,
    "lad": 4,
}

CONFIG_TEMPLATE = """# @package _global_
# Part 2: Round Robin mixed market WITH ZIP, {env_upper} environment
# 13 traders: 7 buyers x 6 sellers (Santa Fe 1991 + ZIP)

experiment:
  name: "p2_rr_mixed_zip_{env}"
  num_rounds: 2

market:
  min_price: 1
  max_price: 2000
  num_tokens: {tokens}
  num_periods: 3
  num_steps: {steps}
  gametype: {gametype}

agents:
  buyer_types: ['ZIC', 'Skeleton', 'Kaplan', 'Ringuette', 'Gamer', 'Perry', 'ZIP']
  seller_types: ['Ledyard', 'BGAN', 'Staecker', 'Jacobson', 'Lin', 'Breton']
"""


def main():
    conf_dir = Path("conf/experiment/p2_tournament/rr")
    conf_dir.mkdir(parents=True, exist_ok=True)

    for env, gametype in ENVS.items():
        config = CONFIG_TEMPLATE.format(
            env=env,
            env_upper=env.upper(),
            gametype=gametype,
            steps=STEPS[env],
            tokens=TOKENS[env],
        )

        filepath = conf_dir / f"p2_rr_mixed_zip_{env}.yaml"
        filepath.write_text(config)
        print(f"Created {filepath}")


if __name__ == "__main__":
    main()
