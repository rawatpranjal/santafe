#!/usr/bin/env python3
"""Generate all paper experiment configs following the naming convention.

Pattern: {part}_{set}_{strategy}_{env}.yaml

Parts: p1, p2, p3, p4
Sets: ctrl (1v7 ZIC), self (self-play), rr (round robin), train (training)
Strategies: zi, zic, zip, gd, kap, skel, ppo, llm
Environments: base, bbbs, bsss, eql, ran, per, shrt, tok, sml, lad
"""

import os
from pathlib import Path

# Environment definitions
ENVS = {
    'base': {'n_buyers': 4, 'n_sellers': 4, 'tokens': 4, 'steps': 100, 'periods': 10, 'gametype': 6453},
    'bbbs': {'n_buyers': 6, 'n_sellers': 2, 'tokens': 4, 'steps': 100, 'periods': 10, 'gametype': 6453},
    'bsss': {'n_buyers': 2, 'n_sellers': 6, 'tokens': 4, 'steps': 100, 'periods': 10, 'gametype': 6453},
    'eql':  {'n_buyers': 4, 'n_sellers': 4, 'tokens': 4, 'steps': 100, 'periods': 10, 'gametype': 5555, 'note': 'symmetric values'},
    'ran':  {'n_buyers': 4, 'n_sellers': 4, 'tokens': 4, 'steps': 100, 'periods': 10, 'gametype': 9999, 'note': 'IID uniform'},
    'per':  {'n_buyers': 4, 'n_sellers': 4, 'tokens': 4, 'steps': 100, 'periods': 1, 'gametype': 6453, 'note': 'single period'},
    'shrt': {'n_buyers': 4, 'n_sellers': 4, 'tokens': 4, 'steps': 20, 'periods': 10, 'gametype': 6453, 'note': 'short steps'},
    'tok':  {'n_buyers': 4, 'n_sellers': 4, 'tokens': 1, 'steps': 100, 'periods': 10, 'gametype': 6453, 'note': '1 token each'},
    'sml':  {'n_buyers': 2, 'n_sellers': 2, 'tokens': 4, 'steps': 100, 'periods': 10, 'gametype': 6453, 'note': '2v2 small'},
    'lad':  {'n_buyers': 4, 'n_sellers': 4, 'tokens': 4, 'steps': 100, 'periods': 10, 'gametype': 6453, 'note': 'ladder/same as base'},
}

# Strategy name mappings
STRATEGY_NAMES = {
    'zi': 'ZI',
    'zic': 'ZIC',
    'zip': 'ZIP',
    'gd': 'GD',
    'kap': 'Kaplan',
    'skel': 'Skeleton',
    'ppo': 'PPO',
    'llm': 'LLM',
}

ENV_NAMES = {
    'base': 'BASE',
    'bbbs': 'BBBS',
    'bsss': 'BSSS',
    'eql': 'EQL',
    'ran': 'RAN',
    'per': 'PER',
    'shrt': 'SHRT',
    'tok': 'TOK',
    'sml': 'SML',
    'lad': 'LAD',
}


def make_agents_list(strategy: str, count: int) -> list[str]:
    """Create list of agent types."""
    return [STRATEGY_NAMES[strategy]] * count


def generate_config(
    name: str,
    description: str,
    env_key: str,
    buyer_types: list[str],
    seller_types: list[str],
    num_rounds: int = 100,
    extra_config: str = ""
) -> str:
    """Generate a YAML config string."""
    env = ENVS[env_key]

    buyers_str = str(buyer_types).replace("'", '"')
    sellers_str = str(seller_types).replace("'", '"')

    config = f"""# @package _global_
# {description}

experiment:
  name: "{name}"
  num_rounds: {num_rounds}

market:
  min_price: 1
  max_price: 1000
  num_tokens: {env['tokens']}
  num_periods: {env['periods']}
  num_steps: {env['steps']}
  gametype: {env['gametype']}

agents:
  buyer_types: {buyers_str}
  seller_types: {sellers_str}
"""
    if extra_config:
        config += extra_config
    return config


def generate_p1_configs(base_path: Path):
    """Generate Part 1: Foundational configs (ZI, ZIC, ZIP self-play)."""
    out_dir = base_path / "p1_foundational"
    out_dir.mkdir(parents=True, exist_ok=True)

    strategies = ['zi', 'zic', 'zip']
    count = 0

    for strat in strategies:
        for env_key in ENVS:
            env = ENVS[env_key]
            name = f"p1_self_{strat}_{env_key}"
            desc = f"Part 1: {STRATEGY_NAMES[strat]} self-play, {ENV_NAMES[env_key]} environment"

            buyers = make_agents_list(strat, env['n_buyers'])
            sellers = make_agents_list(strat, env['n_sellers'])

            config = generate_config(name, desc, env_key, buyers, sellers)

            (out_dir / f"{name}.yaml").write_text(config)
            count += 1

    print(f"  Generated {count} P1 configs")
    return count


def generate_p2_configs(base_path: Path):
    """Generate Part 2: Santa Fe Tournament configs."""
    count = 0

    # Competitor Set 1: Against Control (1 vs 7 ZIC)
    ctrl_dir = base_path / "p2_tournament" / "ctrl"
    ctrl_dir.mkdir(parents=True, exist_ok=True)

    ctrl_strategies = ['skel', 'zip', 'gd', 'kap']
    for strat in ctrl_strategies:
        for env_key in ENVS:
            env = ENVS[env_key]
            name = f"p2_ctrl_{strat}_{env_key}"
            desc = f"Part 2: {STRATEGY_NAMES[strat]} vs 7 ZIC, {ENV_NAMES[env_key]} environment"

            # 1 of strategy X, rest ZIC
            buyers = [STRATEGY_NAMES[strat]] + ['ZIC'] * (env['n_buyers'] - 1)
            sellers = ['ZIC'] * env['n_sellers']

            config = generate_config(name, desc, env_key, buyers, sellers)
            (ctrl_dir / f"{name}.yaml").write_text(config)
            count += 1

    # Competitor Set 2: Self-play
    self_dir = base_path / "p2_tournament" / "self"
    self_dir.mkdir(parents=True, exist_ok=True)

    self_strategies = ['skel', 'zic', 'zip', 'gd', 'kap']
    for strat in self_strategies:
        for env_key in ENVS:
            env = ENVS[env_key]
            name = f"p2_self_{strat}_{env_key}"
            desc = f"Part 2: {STRATEGY_NAMES[strat]} self-play, {ENV_NAMES[env_key]} environment"

            buyers = make_agents_list(strat, env['n_buyers'])
            sellers = make_agents_list(strat, env['n_sellers'])

            config = generate_config(name, desc, env_key, buyers, sellers)
            (self_dir / f"{name}.yaml").write_text(config)
            count += 1

    # Competitor Set 3: Round Robin (Mixed)
    rr_dir = base_path / "p2_tournament" / "rr"
    rr_dir.mkdir(parents=True, exist_ok=True)

    for env_key in ENVS:
        env = ENVS[env_key]
        name = f"p2_rr_mixed_{env_key}"
        desc = f"Part 2: Round Robin mixed market, {ENV_NAMES[env_key]} environment"

        # Mix of all 5 strategies
        all_strategies = ['Skeleton', 'ZIC', 'ZIP', 'GD', 'Kaplan']
        n_total = env['n_buyers'] + env['n_sellers']

        # Distribute strategies across buyers and sellers
        buyers = []
        sellers = []
        for i in range(env['n_buyers']):
            buyers.append(all_strategies[i % len(all_strategies)])
        for i in range(env['n_sellers']):
            sellers.append(all_strategies[(env['n_buyers'] + i) % len(all_strategies)])

        config = generate_config(name, desc, env_key, buyers, sellers)
        (rr_dir / f"{name}.yaml").write_text(config)
        count += 1

    print(f"  Generated {count} P2 configs")
    return count


def generate_p3_configs(base_path: Path):
    """Generate Part 3: PPO configs."""
    count = 0

    # Training configs
    train_dir = base_path / "p3_ppo" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    train_opponents = [
        ('zic', 'ZIC', ['ZIC'] * 7),
        ('skel', 'Skeleton', ['Skeleton'] * 7),
        ('mixed', 'Mixed', ['ZIC', 'ZIP', 'GD', 'Kaplan', 'Skeleton', 'ZIC', 'ZIP']),
    ]

    for opp_key, opp_name, opponents in train_opponents:
        name = f"p3_train_{opp_key}"
        desc = f"Part 3: PPO training vs {opp_name}"

        buyers = ['PPO'] + opponents[:3]
        sellers = opponents[3:]

        extra = """
# Training configuration (Chen et al. 2010)
training:
  total_periods: 7000
  steps_per_period: 25
"""
        config = generate_config(name, desc, 'base', buyers, sellers, num_rounds=1, extra_config=extra)
        (train_dir / f"{name}.yaml").write_text(config)
        count += 1

    # Competitor Set 1: Against Control
    ctrl_dir = base_path / "p3_ppo" / "ctrl"
    ctrl_dir.mkdir(parents=True, exist_ok=True)

    for env_key in ENVS:
        env = ENVS[env_key]
        name = f"p3_ctrl_ppo_{env_key}"
        desc = f"Part 3: PPO vs 7 ZIC, {ENV_NAMES[env_key]} environment"

        buyers = ['PPO'] + ['ZIC'] * (env['n_buyers'] - 1)
        sellers = ['ZIC'] * env['n_sellers']

        config = generate_config(name, desc, env_key, buyers, sellers)
        (ctrl_dir / f"{name}.yaml").write_text(config)
        count += 1

    # Competitor Set 2: Self-play
    self_dir = base_path / "p3_ppo" / "self"
    self_dir.mkdir(parents=True, exist_ok=True)

    for env_key in ENVS:
        env = ENVS[env_key]
        name = f"p3_self_ppo_{env_key}"
        desc = f"Part 3: PPO self-play, {ENV_NAMES[env_key]} environment"

        buyers = ['PPO'] * env['n_buyers']
        sellers = ['PPO'] * env['n_sellers']

        config = generate_config(name, desc, env_key, buyers, sellers)
        (self_dir / f"{name}.yaml").write_text(config)
        count += 1

    # Competitor Set 3: Round Robin
    rr_dir = base_path / "p3_ppo" / "rr"
    rr_dir.mkdir(parents=True, exist_ok=True)

    for env_key in ENVS:
        env = ENVS[env_key]
        name = f"p3_rr_ppo_{env_key}"
        desc = f"Part 3: PPO in mixed market, {ENV_NAMES[env_key]} environment"

        # PPO replaces one legacy trader
        buyers = ['PPO', 'ZIC', 'ZIP', 'GD'][:env['n_buyers']]
        sellers = ['Kaplan', 'Skeleton', 'ZIC', 'ZIP'][:env['n_sellers']]

        config = generate_config(name, desc, env_key, buyers, sellers)
        (rr_dir / f"{name}.yaml").write_text(config)
        count += 1

    print(f"  Generated {count} P3 configs")
    return count


def generate_p4_configs(base_path: Path):
    """Generate Part 4: LLM configs."""
    count = 0

    # Competitor Set 1: Against Control
    ctrl_dir = base_path / "p4_llm" / "ctrl"
    ctrl_dir.mkdir(parents=True, exist_ok=True)

    for env_key in ENVS:
        env = ENVS[env_key]
        name = f"p4_ctrl_llm_{env_key}"
        desc = f"Part 4: LLM vs 7 ZIC, {ENV_NAMES[env_key]} environment"

        buyers = ['LLM'] + ['ZIC'] * (env['n_buyers'] - 1)
        sellers = ['ZIC'] * env['n_sellers']

        config = generate_config(name, desc, env_key, buyers, sellers)
        (ctrl_dir / f"{name}.yaml").write_text(config)
        count += 1

    # Competitor Set 2: Self-play
    self_dir = base_path / "p4_llm" / "self"
    self_dir.mkdir(parents=True, exist_ok=True)

    for env_key in ENVS:
        env = ENVS[env_key]
        name = f"p4_self_llm_{env_key}"
        desc = f"Part 4: LLM self-play, {ENV_NAMES[env_key]} environment"

        buyers = ['LLM'] * env['n_buyers']
        sellers = ['LLM'] * env['n_sellers']

        config = generate_config(name, desc, env_key, buyers, sellers)
        (self_dir / f"{name}.yaml").write_text(config)
        count += 1

    # Competitor Set 3: Round Robin
    rr_dir = base_path / "p4_llm" / "rr"
    rr_dir.mkdir(parents=True, exist_ok=True)

    for env_key in ENVS:
        env = ENVS[env_key]
        name = f"p4_rr_llm_{env_key}"
        desc = f"Part 4: LLM in mixed market, {ENV_NAMES[env_key]} environment"

        # LLM replaces one legacy trader
        buyers = ['LLM', 'ZIC', 'ZIP', 'GD'][:env['n_buyers']]
        sellers = ['Kaplan', 'Skeleton', 'ZIC', 'ZIP'][:env['n_sellers']]

        config = generate_config(name, desc, env_key, buyers, sellers)
        (rr_dir / f"{name}.yaml").write_text(config)
        count += 1

    # Model comparison config
    name = "p4_model_comparison"
    desc = "Part 4: LLM model comparison (GPT-4, GPT-3.5, Claude)"
    extra = """
# Model comparison settings
llm:
  models:
    - name: "gpt-4"
      provider: "openai"
    - name: "gpt-3.5-turbo"
      provider: "openai"
    - name: "claude-3-sonnet"
      provider: "anthropic"
"""
    config = generate_config(name, desc, 'base', ['LLM', 'ZIC', 'ZIC', 'ZIC'], ['ZIC', 'ZIC', 'ZIC', 'ZIC'], extra_config=extra)
    p4_dir = base_path / "p4_llm"
    (p4_dir / f"{name}.yaml").write_text(config)
    count += 1

    print(f"  Generated {count} P4 configs")
    return count


def main():
    """Generate all paper experiment configs."""
    # Find project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    base_path = project_root / "conf" / "experiment"

    print(f"Generating configs in: {base_path}")
    print()

    total = 0
    total += generate_p1_configs(base_path)
    total += generate_p2_configs(base_path)
    total += generate_p3_configs(base_path)
    total += generate_p4_configs(base_path)

    print()
    print(f"Total configs generated: {total}")


if __name__ == "__main__":
    main()
