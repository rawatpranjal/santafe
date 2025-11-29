#!/usr/bin/env python3
"""
Part 1: Foundational Replication Experiments.

Runs ZI, ZIC, ZIP selfplay across 10 environments.
Collects ALL metrics: efficiency, volatility, v-inefficiency, profit dispersion, trades/period.
"""

import logging
import numpy as np
from omegaconf import OmegaConf
from engine.tournament import Tournament

logging.basicConfig(level=logging.WARNING)

# Environment configurations
ENVIRONMENTS = {
    "BASE": {
        "num_buyers": 4, "num_sellers": 4, "num_tokens": 4,
        "num_steps": 100, "gametype": 6453, "num_periods": 10,
        "token_mode": "santafe"
    },
    "BBBS": {
        "num_buyers": 6, "num_sellers": 2, "num_tokens": 4,
        "num_steps": 100, "gametype": 6453, "num_periods": 10,
        "token_mode": "santafe"
    },
    "BSSS": {
        "num_buyers": 2, "num_sellers": 6, "num_tokens": 4,
        "num_steps": 100, "gametype": 6453, "num_periods": 10,
        "token_mode": "santafe"
    },
    "EQL": {
        "num_buyers": 4, "num_sellers": 4, "num_tokens": 4,
        "num_steps": 100, "gametype": 0, "num_periods": 10,
        "token_mode": "santafe"
    },
    "RAN": {
        "num_buyers": 4, "num_sellers": 4, "num_tokens": 4,
        "num_steps": 100, "gametype": 6453, "num_periods": 10,
        "token_mode": "uniform"
    },
    "PER": {
        "num_buyers": 4, "num_sellers": 4, "num_tokens": 4,
        "num_steps": 100, "gametype": 6453, "num_periods": 1,
        "token_mode": "santafe"
    },
    "SHRT": {
        "num_buyers": 4, "num_sellers": 4, "num_tokens": 4,
        "num_steps": 20, "gametype": 6453, "num_periods": 10,
        "token_mode": "santafe"
    },
    "TOK": {
        "num_buyers": 4, "num_sellers": 4, "num_tokens": 1,
        "num_steps": 100, "gametype": 6453, "num_periods": 10,
        "token_mode": "santafe"
    },
    "SML": {
        "num_buyers": 2, "num_sellers": 2, "num_tokens": 4,
        "num_steps": 100, "gametype": 7, "num_periods": 10,
        "token_mode": "santafe"
    },
    "LAD": {
        "num_buyers": 4, "num_sellers": 4, "num_tokens": 4,
        "num_steps": 100, "gametype": 6453, "num_periods": 10,
        "token_mode": "santafe"
    },
}

TRADERS = ["ZI", "ZIC", "ZIP"]
NUM_ROUNDS = 50
NUM_SEEDS = 10

# Metrics to collect
METRICS = ["efficiency", "price_volatility_pct", "v_inefficiency", "profit_dispersion", "trades_per_period"]


def run_experiment(trader_type: str, env_name: str, env_config: dict, seed: int = 42) -> dict:
    """Run selfplay experiment and return all metrics."""

    num_buyers = env_config["num_buyers"]
    num_sellers = env_config["num_sellers"]

    config = OmegaConf.create({
        "experiment": {
            "name": f"part1_{trader_type}_{env_name}_s{seed}",
            "num_rounds": NUM_ROUNDS,
            "rng_seed_values": seed,
            "rng_seed_auction": seed + 1000,
        },
        "market": {
            "gametype": env_config["gametype"],
            "num_tokens": env_config["num_tokens"],
            "min_price": 1,
            "max_price": 1000,
            "num_periods": env_config["num_periods"],
            "num_steps": env_config["num_steps"],
            "token_mode": env_config["token_mode"],
        },
        "agents": {
            "buyer_types": [trader_type] * num_buyers,
            "seller_types": [trader_type] * num_sellers,
        }
    })

    tournament = Tournament(config)
    df = tournament.run()

    # Calculate profit dispersion (RMS of profit deviations)
    # Group by round/period and compute RMS of profit deviations
    profit_dispersions = []
    for (r, p), group in df.groupby(["round", "period"]):
        deviations = group["profit_deviation"].values
        if len(deviations) > 0:
            rms = np.sqrt(np.mean(deviations ** 2))
            profit_dispersions.append(rms)

    # Calculate trades per period
    trades_per_period = df.groupby(["round", "period"])["num_trades"].sum().mean() / 2  # Divide by 2 since each trade counted twice

    return {
        "efficiency": df["efficiency"].mean(),
        "price_volatility_pct": df["price_volatility_pct"].mean(),
        "v_inefficiency": df["v_inefficiency"].mean(),
        "profit_dispersion": np.mean(profit_dispersions) if profit_dispersions else 0,
        "trades_per_period": trades_per_period,
    }


def main():
    # Store all results: results[metric][trader][env] = list of values
    results = {
        metric: {trader: {env: [] for env in ENVIRONMENTS} for trader in TRADERS}
        for metric in METRICS
    }

    total_experiments = len(TRADERS) * len(ENVIRONMENTS) * NUM_SEEDS
    completed = 0

    print(f"Running {total_experiments} experiments...")
    print()

    for trader in TRADERS:
        for env_name, env_config in ENVIRONMENTS.items():
            for seed_idx in range(NUM_SEEDS):
                completed += 1
                seed = 42 + seed_idx * 100

                print(f"[{completed}/{total_experiments}] {trader} × {env_name} (seed {seed_idx+1}/{NUM_SEEDS})...", end=" ", flush=True)

                metrics = run_experiment(trader, env_name, env_config, seed=seed)

                for metric in METRICS:
                    results[metric][trader][env_name].append(metrics[metric])

                print(f"eff={metrics['efficiency']:.0f}%")

    # Compute statistics for each metric
    stats = {
        metric: {trader: {} for trader in TRADERS}
        for metric in METRICS
    }

    for metric in METRICS:
        for trader in TRADERS:
            for env_name in ENVIRONMENTS:
                values = results[metric][trader][env_name]
                stats[metric][trader][env_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

    # Print results
    env_names = list(ENVIRONMENTS.keys())

    metric_labels = {
        "efficiency": ("Efficiency (%)", "{:.0f}±{:.0f}"),
        "price_volatility_pct": ("Price Volatility (%)", "{:.1f}±{:.1f}"),
        "v_inefficiency": ("V-Inefficiency", "{:.1f}±{:.1f}"),
        "profit_dispersion": ("Profit Dispersion", "{:.0f}±{:.0f}"),
        "trades_per_period": ("Trades/Period", "{:.1f}±{:.1f}"),
    }

    print("\n" + "=" * 120)
    print("PART 1 FULL RESULTS")
    print("=" * 120)

    for metric in METRICS:
        label, fmt = metric_labels[metric]
        print(f"\n### Table: {label}\n")
        print("| Trader | " + " | ".join(env_names) + " |")
        print("|--------|" + "|".join("--------" for _ in env_names) + "|")

        for trader in TRADERS:
            row = f"| **{trader}** |"
            for env_name in env_names:
                mean = stats[metric][trader][env_name]["mean"]
                std = stats[metric][trader][env_name]["std"]
                row += " " + fmt.format(mean, std) + " |"
            print(row)


if __name__ == "__main__":
    main()
