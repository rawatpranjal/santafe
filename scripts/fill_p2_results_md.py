#!/usr/bin/env python3
"""Extract Part 2 experiment results and generate markdown tables for results.md.

Reads CSV files from results/p2_* directories and outputs formatted markdown tables
ready to paste into checklists/results.md.

Usage:
    python scripts/fill_p2_results_md.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")

# Environment order
ENVS = ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]

# Strategy name mapping (CSV agent_type -> results.md name)
STRATEGY_MAP = {
    "ZIC1": "ZIC",
    "ZIC": "ZIC",
    "Skeleton": "Skeleton",
    "ZIP1": "ZIP",
    "ZIP": "ZIP",
    "Kaplan": "Kaplan",
    "Ringuette": "Ringuette",
    "GD": "GD",
    "Ledyard": "EL",
    "BGAN": "BGAN",
    "Staecker": "Staecker",
}

# Strategy order for tables
STRATEGY_ORDER = ["ZIC", "Skeleton", "ZIP", "Kaplan", "Ringuette", "GD", "EL", "BGAN", "Staecker"]

# Config name to environment mapping
ENV_MAP = {
    "base": "BASE",
    "bbbs": "BBBS",
    "bsss": "BSSS",
    "eql": "EQL",
    "ran": "RAN",
    "per": "PER",
    "shrt": "SHRT",
    "tok": "TOK",
    "sml": "SML",
    "lad": "LAD",
}

# Config name to strategy mapping
CONFIG_STRATEGY_MAP = {
    "zic": "ZIC",
    "skel": "Skeleton",
    "zip": "ZIP",
    "kap": "Kaplan",
    "ring": "Ringuette",
    "gd": "GD",
    "el": "EL",
    "bgan": "BGAN",
    "staecker": "Staecker",
}


def load_all_results(pattern: str) -> dict[str, pd.DataFrame]:
    """Load all CSVs matching pattern."""
    results = {}
    for result_dir in sorted(RESULTS_DIR.glob(pattern)):
        csv_path = result_dir / "results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            results[result_dir.name] = df
    return results


def parse_config_name(name: str) -> tuple[str, str, str]:
    """Parse config name like p2_self_zip_base -> (category, strategy, env)."""
    parts = name.split("_")
    if len(parts) >= 4:
        category = parts[1]  # self, ctrl, rr
        strategy_code = parts[2]  # zip, zic, kap, etc.
        env_code = parts[3]  # base, bbbs, etc.

        strategy = CONFIG_STRATEGY_MAP.get(strategy_code, strategy_code.upper())
        env = ENV_MAP.get(env_code, env_code.upper())

        return category, strategy, env
    return None, None, None


def generate_selfplay_efficiency_table() -> str:
    """Generate Section 2.2.1 Self-Play Efficiency table."""
    results = load_all_results("p2_self_*")

    # Strategy -> Env -> (mean, std)
    data = defaultdict(dict)

    for name, df in results.items():
        _, strategy, env = parse_config_name(name)
        if strategy and env:
            # Efficiency is per-period, take unique per round then aggregate
            eff_by_round = df.groupby("round")["efficiency"].first()
            mean_eff = eff_by_round.mean()
            std_eff = eff_by_round.std()
            data[strategy][env] = (mean_eff, std_eff)

    # Generate table
    lines = ["#### 2.2.1 Efficiency (%)\n"]
    lines.append("| Strategy | " + " | ".join(ENVS) + " |")
    lines.append("|" + "---|" * (len(ENVS) + 1))

    for strat in STRATEGY_ORDER:
        row = [strat]
        for env in ENVS:
            if strat in data and env in data[strat]:
                mean, std = data[strat][env]
                row.append(f"{mean:.0f}±{std:.0f}")
            else:
                row.append("[[TBD]]")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_selfplay_vineff_table() -> str:
    """Generate Section 2.2.2 V-Inefficiency table."""
    results = load_all_results("p2_self_*")

    data = defaultdict(dict)

    for name, df in results.items():
        _, strategy, env = parse_config_name(name)
        if strategy and env and "v_inefficiency" in df.columns:
            vineff_by_round = df.groupby("round")["v_inefficiency"].first()
            mean_vineff = vineff_by_round.mean()
            std_vineff = vineff_by_round.std()
            data[strategy][env] = (mean_vineff, std_vineff)

    lines = ["#### 2.2.2 V-Inefficiency (Missed Trades)\n"]
    lines.append("| Strategy | " + " | ".join(ENVS) + " |")
    lines.append("|" + "---|" * (len(ENVS) + 1))

    for strat in STRATEGY_ORDER:
        row = [strat]
        for env in ENVS:
            if strat in data and env in data[strat]:
                mean, std = data[strat][env]
                row.append(f"{mean:.0f}±{std:.0f}")
            else:
                row.append("[[TBD]]")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_selfplay_volatility_table() -> str:
    """Generate Section 2.2.3 Price Volatility table."""
    results = load_all_results("p2_self_*")

    data = defaultdict(dict)

    for name, df in results.items():
        _, strategy, env = parse_config_name(name)
        if strategy and env and "price_volatility_pct" in df.columns:
            vol_by_round = df.groupby("round")["price_volatility_pct"].first()
            mean_vol = vol_by_round.mean()
            data[strategy][env] = mean_vol

    lines = ["#### 2.2.3 Price Volatility (%)\n"]
    lines.append("| Strategy | " + " | ".join(ENVS) + " |")
    lines.append("|" + "---|" * (len(ENVS) + 1))

    for strat in STRATEGY_ORDER:
        row = [strat]
        for env in ENVS:
            if strat in data and env in data[strat]:
                row.append(f"{data[strat][env]:.1f}")
            else:
                row.append("[[TBD]]")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_ctrl_efficiency_table() -> str:
    """Generate Section 2.1.1 Invasibility Efficiency table."""
    results = load_all_results("p2_ctrl_*")

    data = defaultdict(dict)

    for name, df in results.items():
        _, strategy, env = parse_config_name(name)
        if strategy and env:
            eff_by_round = df.groupby("round")["efficiency"].first()
            mean_eff = eff_by_round.mean()
            std_eff = eff_by_round.std()
            data[strategy][env] = (mean_eff, std_eff)

    lines = ["#### 2.1.1 Efficiency (%)\n"]
    lines.append("| Strategy | " + " | ".join(ENVS) + " |")
    lines.append("|" + "---|" * (len(ENVS) + 1))

    # Ctrl doesn't include ZIC (it's the control)
    ctrl_strategies = [s for s in STRATEGY_ORDER if s != "ZIC"]

    for strat in ctrl_strategies:
        row = [strat]
        for env in ENVS:
            if strat in data and env in data[strat]:
                mean, std = data[strat][env]
                row.append(f"{mean:.0f}±{std:.0f}")
            else:
                row.append("[[TBD]]")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_ctrl_profit_ratio_table() -> str:
    """Generate Section 2.1.2 Profit Ratio table."""
    results = load_all_results("p2_ctrl_*")

    data = defaultdict(dict)

    for name, df in results.items():
        _, strategy, env = parse_config_name(name)
        if strategy and env:
            # Map agent types
            df["mapped_type"] = df["agent_type"].map(lambda x: STRATEGY_MAP.get(x, x))

            # Get focal strategy profit (not ZIC)
            focal_df = df[df["mapped_type"] == strategy]
            zic_df = df[df["mapped_type"] == "ZIC"]

            if len(focal_df) > 0 and len(zic_df) > 0:
                focal_profit = focal_df.groupby("round")["period_profit"].sum().mean()
                zic_profit = zic_df.groupby("round")["period_profit"].sum().mean()

                if zic_profit > 0:
                    ratio = focal_profit / zic_profit
                    data[strategy][env] = ratio
                elif zic_profit < 0:
                    data[strategy][env] = "N/A"  # ZIC has negative profit

    lines = ["#### 2.1.2 Profit Ratio (Challenger / ZIC)\n"]
    lines.append("*Ratio > 1.0 means exploitation of ZIC.*\n")
    lines.append("| Strategy | " + " | ".join(ENVS) + " |")
    lines.append("|" + "---|" * (len(ENVS) + 1))

    ctrl_strategies = [s for s in STRATEGY_ORDER if s != "ZIC"]

    for strat in ctrl_strategies:
        row = [strat]
        for env in ENVS:
            if strat in data and env in data[strat]:
                val = data[strat][env]
                if val == "N/A":
                    row.append("N/A")
                else:
                    row.append(f"{val:.2f}x")
            else:
                row.append("[[TBD]]")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_ctrl_volatility_table() -> str:
    """Generate Section 2.1.3 Price Volatility table."""
    results = load_all_results("p2_ctrl_*")

    data = defaultdict(dict)

    for name, df in results.items():
        _, strategy, env = parse_config_name(name)
        if strategy and env and "price_volatility_pct" in df.columns:
            vol_by_round = df.groupby("round")["price_volatility_pct"].first()
            mean_vol = vol_by_round.mean()
            data[strategy][env] = mean_vol

    lines = ["#### 2.1.3 Price Volatility (%)\n"]
    lines.append("| Strategy | " + " | ".join(ENVS) + " |")
    lines.append("|" + "---|" * (len(ENVS) + 1))

    ctrl_strategies = [s for s in STRATEGY_ORDER if s != "ZIC"]

    for strat in ctrl_strategies:
        row = [strat]
        for env in ENVS:
            if strat in data and env in data[strat]:
                row.append(f"{data[strat][env]:.1f}")
            else:
                row.append("[[TBD]]")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_roundrobin_profit_table() -> str:
    """Generate Section 2.3.1 Round-Robin Profit table."""
    results = load_all_results("p2_rr_*")

    # Env -> Strategy -> profit
    data = defaultdict(dict)

    for name, df in results.items():
        parts = name.split("_")
        if len(parts) >= 4:
            env_code = parts[3]
            env = ENV_MAP.get(env_code, env_code.upper())

            df["mapped_type"] = df["agent_type"].map(lambda x: STRATEGY_MAP.get(x, x))

            for agent_type in df["mapped_type"].unique():
                agent_df = df[df["mapped_type"] == agent_type]
                total_profit = agent_df["period_profit"].sum()
                num_rounds = agent_df["round"].nunique()
                avg_profit = total_profit / num_rounds if num_rounds > 0 else 0

                mapped_name = agent_type
                data[env][mapped_name] = avg_profit

    lines = ["#### 2.3.1 Profit by Strategy (mean per round)\n"]
    lines.append("| Env | " + " | ".join(STRATEGY_ORDER) + " |")
    lines.append("|" + "---|" * (len(STRATEGY_ORDER) + 1))

    for env in ENVS:
        row = [env]
        for strat in STRATEGY_ORDER:
            if env in data and strat in data[env]:
                profit = data[env][strat]
                if abs(profit) >= 1000:
                    row.append(f"{profit/1000:.0f}k")
                else:
                    row.append(f"{profit:.0f}")
            else:
                row.append("[[TBD]]")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_roundrobin_summary() -> str:
    """Generate Section 2.3.3 Tournament Summary."""
    results = load_all_results("p2_rr_*")

    # Strategy -> list of ranks across envs
    ranks_by_strategy = defaultdict(list)
    wins_by_strategy = defaultdict(int)

    for name, df in results.items():
        parts = name.split("_")
        if len(parts) >= 4:
            env_code = parts[3]
            env = ENV_MAP.get(env_code, env_code.upper())

            df["mapped_type"] = df["agent_type"].map(lambda x: STRATEGY_MAP.get(x, x))

            # Calculate total profit per strategy
            profits = {}
            for agent_type in df["mapped_type"].unique():
                agent_df = df[df["mapped_type"] == agent_type]
                total_profit = agent_df["period_profit"].sum()
                profits[agent_type] = total_profit

            # Rank strategies
            sorted_strats = sorted(profits.keys(), key=lambda x: profits[x], reverse=True)
            for i, strat in enumerate(sorted_strats):
                ranks_by_strategy[strat].append(i + 1)
                if i == 0:
                    wins_by_strategy[strat] += 1

    lines = ["#### 2.3.3 Tournament Summary\n"]
    lines.append("| Strategy | Avg Rank | Wins | Envs |")
    lines.append("|---|---|---|---|")

    # Sort by average rank
    avg_ranks = {s: np.mean(r) if r else float("inf") for s, r in ranks_by_strategy.items()}
    sorted_strats = sorted(avg_ranks.keys(), key=lambda x: avg_ranks[x])

    for strat in sorted_strats:
        ranks = ranks_by_strategy[strat]
        if ranks:
            avg_rank = np.mean(ranks)
            wins = wins_by_strategy[strat]
            envs = len(ranks)
            lines.append(f"| {strat} | {avg_rank:.2f} | {wins} | {envs} |")

    return "\n".join(lines)


def generate_evolutionary_tables() -> str:
    """Generate Section 2.4 Evolutionary tables from JSON files."""
    # Find evolution result files
    evo_files = list(RESULTS_DIR.glob("evolution_v3_seed*.json"))
    if not evo_files:
        evo_files = list(RESULTS_DIR.glob("evolution_*.json"))

    if not evo_files:
        return "#### 2.4 Evolutionary Tournament\n\n*No evolution data found.*\n"

    # Load and aggregate
    all_generations = defaultdict(lambda: defaultdict(list))
    final_distributions = []

    for evo_file in sorted(evo_files)[:10]:  # Limit to 10 seeds
        try:
            with open(evo_file) as f:
                data = json.load(f)

            generations = data.get("generations", [])
            for gen_data in generations:
                gen_num = gen_data.get("generation", 0)
                counts = gen_data.get("strategy_counts", {})
                total = sum(counts.values())

                for strat, count in counts.items():
                    pct = 100 * count / total if total > 0 else 0
                    all_generations[gen_num][strat].append(pct)

            # Get final distribution
            if generations:
                final = generations[-1].get("strategy_counts", {})
                final_distributions.append(final)
        except Exception as e:
            print(f"Error loading {evo_file}: {e}")
            continue

    if not all_generations:
        return "#### 2.4 Evolutionary Tournament\n\n*Could not parse evolution data.*\n"

    lines = ["#### 2.4.1 Population Share Over Generations\n"]

    # Get available strategies from gen 0
    strategies = list(all_generations.get(0, {}).keys())

    lines.append("| Generation | " + " | ".join(strategies) + " |")
    lines.append("|" + "---|" * (len(strategies) + 1))

    target_gens = [0, 10, 20, 50, 100]
    available_gens = sorted(all_generations.keys())

    for target in target_gens:
        # Find closest available generation
        closest = min(available_gens, key=lambda x: abs(x - target)) if available_gens else None

        if closest is not None and closest in all_generations:
            row = [f"{closest}"]
            for strat in strategies:
                values = all_generations[closest].get(strat, [])
                if values:
                    mean_pct = np.mean(values)
                    row.append(f"{mean_pct:.1f}%")
                else:
                    row.append("0%")
            lines.append("| " + " | ".join(row) + " |")
        else:
            row = [f"{target}"] + ["[[TBD]]"] * len(strategies)
            lines.append("| " + " | ".join(row) + " |")

    # Final population summary
    lines.append("\n#### 2.4.2 Final Population (Last Generation)\n")

    if final_distributions:
        # Average final distribution
        avg_final = defaultdict(float)
        for dist in final_distributions:
            total = sum(dist.values())
            for strat, count in dist.items():
                avg_final[strat] += (100 * count / total if total > 0 else 0) / len(
                    final_distributions
                )

        sorted_final = sorted(avg_final.items(), key=lambda x: x[1], reverse=True)

        lines.append("| Strategy | Population Share | Classification |")
        lines.append("|---|---|---|")

        for i, (strat, pct) in enumerate(sorted_final):
            if pct > 0:
                classification = "ESS" if i == 0 and pct > 30 else ""
                lines.append(f"| {strat} | {pct:.1f}% | {classification} |")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("Part 2 Results Extraction for results.md")
    print("=" * 60)

    print("\n## Section 2.1: Invasibility (1 vs 7 ZIC)\n")
    print(generate_ctrl_efficiency_table())
    print()
    print(generate_ctrl_profit_ratio_table())
    print()
    print(generate_ctrl_volatility_table())

    print("\n## Section 2.2: Self-Play (8 Identical Agents)\n")
    print(generate_selfplay_efficiency_table())
    print()
    print(generate_selfplay_vineff_table())
    print()
    print(generate_selfplay_volatility_table())

    print("\n## Section 2.3: Round-Robin Tournament\n")
    print(generate_roundrobin_profit_table())
    print()
    print(generate_roundrobin_summary())

    print("\n## Section 2.4: Evolutionary Tournament\n")
    print(generate_evolutionary_tables())


if __name__ == "__main__":
    main()
