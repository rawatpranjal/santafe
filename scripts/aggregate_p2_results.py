#!/usr/bin/env python3
"""Aggregate Part 2 experiment results for results.md tables."""
import pandas as pd
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path("results")

# Environment mapping from config names
ENV_MAP = {
    "base": "BASE", "bbbs": "BBBS", "bsss": "BSSS", "eql": "EQL",
    "ran": "RAN", "per": "PER", "shrt": "SHRT", "tok": "TOK",
    "sml": "SML", "lad": "LAD"
}

STRATEGY_MAP = {
    "kap": "Kaplan", "skel": "Skeleton", "ske": "Skeleton", "zic": "ZIC", "zip": "ZIP", "gd": "GD"
}

def parse_experiment_name(name: str) -> tuple:
    """Parse experiment name like p2_self_zip_base -> (category, strategy, env)."""
    parts = name.split("_")
    if len(parts) >= 4:
        category = parts[1]  # self, ctrl, rr
        strategy = parts[2]  # zip, zic, kap, etc.
        env = parts[3]       # base, bbbs, etc.
        return category, strategy, env
    return None, None, None

def load_results(pattern: str) -> dict:
    """Load all results matching pattern."""
    results = {}
    for result_dir in sorted(RESULTS_DIR.glob(pattern)):
        csv_path = result_dir / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            results[result_dir.name] = df
    return results

def aggregate_self_play():
    """Aggregate self-play results (8 same-type traders)."""
    results = load_results("p2_self_*")

    # Strategy -> Env -> efficiency
    table = defaultdict(dict)

    for name, df in results.items():
        _, strategy, env = parse_experiment_name(name)
        if strategy and env:
            mean_eff = df["efficiency"].mean()
            std_eff = df["efficiency"].std()
            strat_name = STRATEGY_MAP.get(strategy, strategy.upper())
            env_name = ENV_MAP.get(env, env.upper())
            table[strat_name][env_name] = f"{mean_eff:.1f}±{std_eff:.1f}"

    return table

def aggregate_control():
    """Aggregate control results (1 strategy vs 7 ZIC)."""
    results = load_results("p2_ctrl_*")

    # Strategy -> Env -> (efficiency, profit_deviation)
    table = defaultdict(dict)

    for name, df in results.items():
        _, strategy, env = parse_experiment_name(name)
        if strategy and env:
            mean_eff = df["efficiency"].mean()
            std_eff = df["efficiency"].std()

            # Get the focal strategy's profit deviation
            # The focal strategy is the first agent (not ZIC)
            focal_type = STRATEGY_MAP.get(strategy, strategy.upper())
            focal_df = df[df["agent_type"] == focal_type]
            if len(focal_df) > 0:
                profit_dev = focal_df["profit_deviation"].mean()
            else:
                profit_dev = 0

            env_name = ENV_MAP.get(env, env.upper())
            table[focal_type][env_name] = {
                "eff": f"{mean_eff:.1f}±{std_eff:.1f}",
                "dev": f"{profit_dev:+.0f}"
            }

    return table

def aggregate_round_robin():
    """Aggregate round-robin results (mixed market)."""
    results = load_results("p2_rr_*")

    # Env -> {strategy -> avg_profit}
    table = defaultdict(dict)

    for name, df in results.items():
        parts = name.split("_")
        if len(parts) >= 4:
            env = parts[3]  # base, bbbs, etc.
            env_name = ENV_MAP.get(env, env.upper())

            # Get efficiency
            mean_eff = df["efficiency"].mean()

            # Get profit by strategy type
            for agent_type in df["agent_type"].unique():
                agent_df = df[df["agent_type"] == agent_type]
                avg_profit = agent_df["period_profit"].sum() / agent_df["round"].nunique()
                table[env_name][agent_type] = avg_profit

            table[env_name]["_efficiency"] = mean_eff

    return table

def print_self_play_table():
    """Print self-play table for results.md."""
    table = aggregate_self_play()
    envs = ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]
    strategies = ["Skeleton", "ZIC", "ZIP", "Kaplan"]

    print("\n### Table 2.2: Self-Play Efficiency (%)\n")
    print("| Strategy |", " | ".join(envs), "|")
    print("|" + "---|" * (len(envs) + 1))

    for strat in strategies:
        row = [strat]
        for env in envs:
            val = table.get(strat, {}).get(env, "⬜")
            row.append(val)
        print("| " + " | ".join(row) + " |")

def print_control_table():
    """Print control table for results.md."""
    table = aggregate_control()
    envs = ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]
    strategies = ["Skeleton", "ZIP", "Kaplan"]  # GD excluded

    print("\n### Table 2.1: Against Control (Efficiency %)\n")
    print("| Strategy |", " | ".join(envs), "|")
    print("|" + "---|" * (len(envs) + 1))

    for strat in strategies:
        row = [strat]
        for env in envs:
            data = table.get(strat, {}).get(env, {})
            if isinstance(data, dict):
                val = data.get("eff", "⬜")
            else:
                val = "⬜"
            row.append(val)
        print("| " + " | ".join(row) + " |")

def print_round_robin_table():
    """Print round-robin table for results.md."""
    table = aggregate_round_robin()
    envs = ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]

    print("\n### Table 2.3: Round Robin Tournament\n")
    print("| Environment | Efficiency | Top Strategy | 2nd | 3rd |")
    print("|---|---|---|---|---|")

    for env in envs:
        data = table.get(env, {})
        if data:
            eff = data.get("_efficiency", 0)
            # Sort strategies by profit
            profits = [(k, v) for k, v in data.items() if not k.startswith("_")]
            profits.sort(key=lambda x: x[1], reverse=True)

            if len(profits) >= 3:
                top = f"{profits[0][0]} ({profits[0][1]:.0f})"
                second = f"{profits[1][0]} ({profits[1][1]:.0f})"
                third = f"{profits[2][0]} ({profits[2][1]:.0f})"
            else:
                top, second, third = "⬜", "⬜", "⬜"

            print(f"| {env} | {eff:.1f}% | {top} | {second} | {third} |")
        else:
            print(f"| {env} | ⬜ | ⬜ | ⬜ | ⬜ |")

if __name__ == "__main__":
    print("=" * 60)
    print("Part 2 Experiment Results Aggregation")
    print("=" * 60)

    # Count results
    self_count = len(list(RESULTS_DIR.glob("p2_self_*")))
    ctrl_count = len(list(RESULTS_DIR.glob("p2_ctrl_*")))
    rr_count = len(list(RESULTS_DIR.glob("p2_rr_*")))

    print(f"\nFound: {self_count} self-play, {ctrl_count} control, {rr_count} round-robin\n")

    print_self_play_table()
    print_control_table()
    print_round_robin_table()
