#!/usr/bin/env python3
"""
Generate all tournament configuration files for the all-vs-all tournament.

Creates 91 experiment configs:
- 10 pure markets (homogeneous populations)
- 45 pairwise combinations
- 10 mixed populations (including background sweeps)
- 10 one-vs-seven ZIC tests
- 10 one-vs-seven mixed tests
- 6 asymmetric condition tests
"""

from pathlib import Path
from itertools import combinations

# Base directory
CONF_DIR = Path("conf/experiment/tournament")

# All available traders
ALL_TRADERS = ["ZI", "ZIC", "Kaplan", "ZIP", "GD", "ZI2", "Skeleton", "Lin", "Jacobson", "Perry"]
CORE_TRADERS = ["ZI", "ZIC", "Kaplan", "ZIP", "GD", "ZI2"]

# Standard tournament setup
NUM_BUYERS = 8
NUM_SELLERS = 8
NUM_ROUNDS = 100


def create_pure_configs():
    """Create 10 pure market configs (all same trader type)."""
    print("Creating pure market configs...")
    pure_dir = CONF_DIR / "pure"

    for trader in ALL_TRADERS:
        config_name = f"pure_{trader.lower()}.yaml"
        config_path = pure_dir / config_name

        content = f"""# @package _global_
# Pure {trader} market - homogeneous population
experiment:
  name: "pure_{trader.lower()}"
  num_rounds: {NUM_ROUNDS}

agents:
  buyer_types: {[trader] * NUM_BUYERS}
  seller_types: {[trader] * NUM_SELLERS}
"""

        with open(config_path, 'w') as f:
            f.write(content)

        print(f"  Created {config_name}")

    print(f"✓ Created {len(ALL_TRADERS)} pure market configs\n")


def create_pairwise_configs():
    """Create 45 pairwise tournament configs (all 2-trader combinations)."""
    print("Creating pairwise tournament configs...")
    pairwise_dir = CONF_DIR / "pairwise"

    pairs = list(combinations(ALL_TRADERS, 2))

    for trader_a, trader_b in pairs:
        config_name = f"{trader_a.lower()}_vs_{trader_b.lower()}.yaml"
        config_path = pairwise_dir / config_name

        # 4 of each type for buyers and sellers
        buyer_types = [trader_a] * 4 + [trader_b] * 4
        seller_types = [trader_a] * 4 + [trader_b] * 4

        content = f"""# @package _global_
# Pairwise: {trader_a} vs {trader_b}
experiment:
  name: "pairwise_{trader_a.lower()}_vs_{trader_b.lower()}"
  num_rounds: {NUM_ROUNDS}

agents:
  buyer_types: {buyer_types}
  seller_types: {seller_types}
"""

        with open(config_path, 'w') as f:
            f.write(content)

        print(f"  Created {config_name}")

    print(f"✓ Created {len(pairs)} pairwise configs\n")


def create_mixed_configs():
    """Create 10 mixed population configs."""
    print("Creating mixed population configs...")
    mixed_dir = CONF_DIR / "mixed"

    configs = []

    # 1. Equal mix of 5 core traders (excluding ZI)
    # Each trader gets ~1-2 slots: distribute 8 buyers/sellers
    core_5 = ["ZIC", "Kaplan", "ZIP", "GD", "ZI2"]
    buyer_types_core5 = ["ZIC", "ZIC", "Kaplan", "ZIP", "GD", "GD", "ZI2", "ZI2"]
    seller_types_core5 = ["ZIC", "Kaplan", "Kaplan", "ZIP", "ZIP", "GD", "ZI2", "ZI2"]

    configs.append(("equal_mix_5core.yaml", "equal_mix_5core", buyer_types_core5, seller_types_core5))

    # 2. Equal mix of all 10 traders (can't fit perfectly, some get 1, some get 0)
    buyer_types_all10 = ["ZI", "ZIC", "Kaplan", "ZIP", "GD", "ZI2", "Skeleton", "Lin"]
    seller_types_all10 = ["ZI", "ZIC", "Kaplan", "ZIP", "GD", "ZI2", "Jacobson", "Perry"]

    configs.append(("equal_mix_10all.yaml", "equal_mix_10all", buyer_types_all10, seller_types_all10))

    # 3-9. Kaplan background sweep: 0%, 10%, 25%, 50%, 75%, 90%, 100%
    background_base = ["ZIC", "ZIP", "GD", "ZI2"]

    kaplan_percentages = [
        (0, "00pct"),
        (1, "10pct"),   # 1 Kaplan, 7 background
        (2, "25pct"),   # 2 Kaplan, 6 background
        (4, "50pct"),   # 4 Kaplan, 4 background
        (6, "75pct"),   # 6 Kaplan, 2 background
        (7, "90pct"),   # 7 Kaplan, 1 background
    ]

    for num_kaplan, pct_label in kaplan_percentages:
        num_background = 8 - num_kaplan

        # Distribute background traders evenly
        background_types = []
        for i in range(num_background):
            background_types.append(background_base[i % len(background_base)])

        buyer_types = ["Kaplan"] * num_kaplan + background_types
        seller_types = ["Kaplan"] * num_kaplan + background_types

        config_name = f"kaplan_background_{pct_label}.yaml"
        exp_name = f"kaplan_background_{pct_label}"

        configs.append((config_name, exp_name, buyer_types, seller_types))

    # Write all configs
    for config_name, exp_name, buyer_types, seller_types in configs:
        config_path = mixed_dir / config_name

        content = f"""# @package _global_
# Mixed population: {exp_name}
experiment:
  name: "{exp_name}"
  num_rounds: {NUM_ROUNDS}

agents:
  buyer_types: {buyer_types}
  seller_types: {seller_types}
"""

        with open(config_path, 'w') as f:
            f.write(content)

        print(f"  Created {config_name}")

    print(f"✓ Created {len(configs)} mixed population configs\n")


def create_one_v_seven_configs():
    """Create 20 one-vs-seven configs (10 vs ZIC, 10 vs mixed)."""
    print("Creating one-vs-seven configs...")
    one_v_seven_dir = CONF_DIR / "one_v_seven"

    # Background for mixed: distribute among ZIC, ZIP, GD
    mixed_background = ["ZIC", "ZIC", "ZIC", "ZIP", "ZIP", "GD", "GD"]

    for trader in ALL_TRADERS:
        # 1. One trader vs 7 ZIC
        config_name_zic = f"{trader.lower()}_1v7_zic.yaml"
        config_path_zic = one_v_seven_dir / config_name_zic

        buyer_types_zic = [trader] + ["ZIC"] * 7
        seller_types_zic = [trader] + ["ZIC"] * 7

        content_zic = f"""# @package _global_
# One {trader} vs 7 ZIC (parasitic test)
experiment:
  name: "{trader.lower()}_1v7_zic"
  num_rounds: {NUM_ROUNDS}

agents:
  buyer_types: {buyer_types_zic}
  seller_types: {seller_types_zic}
"""

        with open(config_path_zic, 'w') as f:
            f.write(content_zic)

        print(f"  Created {config_name_zic}")

        # 2. One trader vs 7 mixed
        config_name_mixed = f"{trader.lower()}_1v7_mixed.yaml"
        config_path_mixed = one_v_seven_dir / config_name_mixed

        buyer_types_mixed = [trader] + mixed_background
        seller_types_mixed = [trader] + mixed_background

        content_mixed = f"""# @package _global_
# One {trader} vs 7 mixed traders (competitive test)
experiment:
  name: "{trader.lower()}_1v7_mixed"
  num_rounds: {NUM_ROUNDS}

agents:
  buyer_types: {buyer_types_mixed}
  seller_types: {seller_types_mixed}
"""

        with open(config_path_mixed, 'w') as f:
            f.write(content_mixed)

        print(f"  Created {config_name_mixed}")

    print(f"✓ Created {len(ALL_TRADERS) * 2} one-vs-seven configs\n")


def create_asymmetric_configs():
    """Create 6 asymmetric condition configs."""
    print("Creating asymmetric condition configs...")
    asym_dir = CONF_DIR / "asymmetric"

    # Use ZIC as the baseline trader for asymmetric tests
    configs = []

    # 1. 3 buyers vs 5 sellers (supply imbalance)
    configs.append({
        "name": "asym_3v5_buyers.yaml",
        "exp_name": "asym_3v5_buyers",
        "buyer_types": ["ZIC", "ZIC", "ZIC"],
        "seller_types": ["ZIC", "ZIC", "ZIC", "ZIC", "ZIC"],
        "comment": "3 buyers vs 5 sellers - supply imbalance"
    })

    # 2. 5 buyers vs 3 sellers (demand imbalance)
    configs.append({
        "name": "asym_5v3_buyers.yaml",
        "exp_name": "asym_5v3_buyers",
        "buyer_types": ["ZIC", "ZIC", "ZIC", "ZIC", "ZIC"],
        "seller_types": ["ZIC", "ZIC", "ZIC"],
        "comment": "5 buyers vs 3 sellers - demand imbalance"
    })

    # For token imbalances, we need to check if the config supports this
    # Looking at the base config, num_tokens is a global setting
    # We'll create configs but note they may need engine support

    # 3. Token heavy buyers (note: may need engine modification)
    configs.append({
        "name": "asym_token_heavy_buyers.yaml",
        "exp_name": "asym_token_heavy_buyers",
        "buyer_types": ["ZIC"] * 8,
        "seller_types": ["ZIC"] * 8,
        "comment": "Equal count but token imbalance (needs engine support)"
    })

    # 4. Token heavy sellers
    configs.append({
        "name": "asym_token_heavy_sellers.yaml",
        "exp_name": "asym_token_heavy_sellers",
        "buyer_types": ["ZIC"] * 8,
        "seller_types": ["ZIC"] * 8,
        "comment": "Equal count but token imbalance (needs engine support)"
    })

    # 5. Mixed traders with asymmetric counts
    configs.append({
        "name": "asym_mixed_3v5.yaml",
        "exp_name": "asym_mixed_3v5",
        "buyer_types": ["ZIC", "Kaplan", "ZIP"],
        "seller_types": ["ZIC", "ZIP", "GD", "Kaplan", "ZI2"],
        "comment": "Mixed traders, 3v5 asymmetric"
    })

    # 6. Mixed traders with asymmetric counts (reversed)
    configs.append({
        "name": "asym_mixed_5v3.yaml",
        "exp_name": "asym_mixed_5v3",
        "buyer_types": ["ZIC", "Kaplan", "ZIP", "GD", "ZI2"],
        "seller_types": ["ZIC", "ZIP", "Kaplan"],
        "comment": "Mixed traders, 5v3 asymmetric"
    })

    for cfg in configs:
        config_path = asym_dir / cfg["name"]

        content = f"""# @package _global_
# {cfg['comment']}
experiment:
  name: "{cfg['exp_name']}"
  num_rounds: {NUM_ROUNDS}

agents:
  buyer_types: {cfg['buyer_types']}
  seller_types: {cfg['seller_types']}
"""

        with open(config_path, 'w') as f:
            f.write(content)

        print(f"  Created {cfg['name']}")

    print(f"✓ Created {len(configs)} asymmetric configs\n")


def main():
    """Generate all tournament configs."""
    print("=" * 60)
    print("GENERATING TOURNAMENT CONFIGURATION FILES")
    print("=" * 60)
    print()

    # Ensure directories exist
    for subdir in ["pure", "pairwise", "mixed", "one_v_seven", "asymmetric"]:
        (CONF_DIR / subdir).mkdir(parents=True, exist_ok=True)

    # Generate all configs
    create_pure_configs()
    create_pairwise_configs()
    create_mixed_configs()
    create_one_v_seven_configs()
    create_asymmetric_configs()

    # Summary
    total_configs = (
        len(ALL_TRADERS) +  # Pure
        len(list(combinations(ALL_TRADERS, 2))) +  # Pairwise
        8 +  # Mixed (2 equal + 6 kaplan background)
        len(ALL_TRADERS) * 2 +  # One-vs-seven
        6  # Asymmetric
    )

    print("=" * 60)
    print(f"✓ TOTAL CONFIGS CREATED: {total_configs}")
    print("=" * 60)
    print()
    print("Breakdown:")
    print(f"  • Pure markets:       {len(ALL_TRADERS)}")
    print(f"  • Pairwise combos:    {len(list(combinations(ALL_TRADERS, 2)))}")
    print(f"  • Mixed populations:  8")
    print(f"  • One-vs-seven tests: {len(ALL_TRADERS) * 2}")
    print(f"  • Asymmetric tests:   6")
    print()


if __name__ == "__main__":
    main()
