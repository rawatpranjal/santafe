#!/usr/bin/env python3
"""Aggregate Part 2 Santa Fe experiment results for results.md."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Santa Fe trader roster (12 traders)
TRADERS = {
    "bgan": "BGAN",
    "breton": "Breton",
    "el": "EL",
    "gamer": "Gamer",
    "jacobson": "Jacobson",
    "kap": "Kaplan",
    "lin": "Lin",
    "perry": "Perry",
    "ring": "Ringuette",
    "skel": "Skeleton",
    "staecker": "Staecker",
    "zic": "ZIC",
}

ENVS = ["base", "bbbs", "bsss", "eql", "ran", "per", "shrt", "tok", "sml", "lad"]


def aggregate_self_play():
    """Aggregate self-play results."""
    results = {}

    for trader_short, trader_name in TRADERS.items():
        results[trader_name] = {}

        for env in ENVS:
            csv_path = Path(f"results/p2_self_{trader_short}_{env}/results.csv")
            if not csv_path.exists():
                print(f"Missing: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            # Get per-period metrics (one row per period)
            period_data = df.groupby(["round", "period"]).first().reset_index()

            results[trader_name][env.upper()] = {
                "efficiency": float(period_data["efficiency"].mean()),
                "efficiency_std": float(period_data["efficiency"].std()),
                "v_inefficiency": float(period_data["v_inefficiency"].mean()),
                "volatility": float(period_data["smiths_alpha"].mean()),
                "trades": float(
                    period_data["num_trades"].sum() / len(period_data["round"].unique())
                ),
                "profit_dispersion": float(period_data["profit_dispersion"].mean()),
                "rmsd": float(period_data["rmsd"].mean()),
                "trades_per_period": float(period_data["num_trades"].mean()),
            }

    return results


def aggregate_control():
    """Aggregate control (1 vs 7 ZIC) results."""
    results = {}

    for trader_short, trader_name in TRADERS.items():
        if trader_short == "zic":
            continue  # ZIC is the control, skip

        results[trader_name] = {}

        for env in ENVS:
            csv_path = Path(f"results/p2_ctrl_{trader_short}_{env}/results.csv")
            if not csv_path.exists():
                print(f"Missing: {csv_path}")
                continue

            df = pd.read_csv(csv_path)

            # Get per-period metrics
            period_data = df.groupby(["round", "period"]).first().reset_index()

            # Get test trader profit (agent_id=1 is always the test trader)
            test_trader = df[df["agent_id"] == 1]
            test_profit = test_trader.groupby("round")["period_profit"].sum().mean()

            # Get ZIC baseline profit (average of agents 2-8)
            zic_traders = df[df["agent_id"] > 1]
            zic_profit = zic_traders.groupby("round")["period_profit"].sum().mean() / 7

            results[trader_name][env.upper()] = {
                "efficiency": float(period_data["efficiency"].mean()),
                "test_profit": float(test_profit),
                "zic_profit": float(zic_profit),
                "profit_ratio": float(test_profit / zic_profit) if zic_profit > 0 else 0,
            }

    return results


def print_self_play_table(results: dict):
    """Print self-play efficiency table."""
    print("\n## 2.2.1 Self-Play Efficiency (Allocative Efficiency %)\n")
    print("| Strategy | BASE | BBBS | BSSS | EQL | RAN | PER | SHRT | TOK | SML | LAD | Mean |")
    print("|----------|------|------|------|-----|-----|-----|------|-----|-----|-----|------|")

    for trader in sorted(results.keys()):
        row = [trader]
        values = []
        for env in ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]:
            if env in results[trader]:
                val = results[trader][env]["efficiency"]
                row.append(f"{val:.1f}")
                values.append(val)
            else:
                row.append("-")

        mean = np.mean(values) if values else 0
        row.append(f"{mean:.1f}")
        print("| " + " | ".join(row) + " |")


def print_vineff_table(results: dict):
    """Print v-inefficiency table."""
    print("\n## 2.2.2 V-Inefficiency\n")
    print("| Strategy | BASE | BBBS | BSSS | EQL | RAN | PER | SHRT | TOK | SML | LAD | Mean |")
    print("|----------|------|------|------|-----|-----|-----|------|-----|-----|-----|------|")

    for trader in sorted(results.keys()):
        row = [trader]
        values = []
        for env in ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]:
            if env in results[trader]:
                val = results[trader][env]["v_inefficiency"]
                row.append(f"{val:.1f}")
                values.append(val)
            else:
                row.append("-")

        mean = np.mean(values) if values else 0
        row.append(f"{mean:.1f}")
        print("| " + " | ".join(row) + " |")


def print_volatility_table(results: dict):
    """Print price volatility table."""
    print("\n## 2.2.3 Price Volatility (Smith's Alpha)\n")
    print("| Strategy | BASE | BBBS | BSSS | EQL | RAN | PER | SHRT | TOK | SML | LAD | Mean |")
    print("|----------|------|------|------|-----|-----|-----|------|-----|-----|-----|------|")

    for trader in sorted(results.keys()):
        row = [trader]
        values = []
        for env in ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]:
            if env in results[trader]:
                val = results[trader][env]["volatility"]
                row.append(f"{val:.1f}")
                values.append(val)
            else:
                row.append("-")

        mean = np.mean(values) if values else 0
        row.append(f"{mean:.1f}")
        print("| " + " | ".join(row) + " |")


def print_control_table(results: dict):
    """Print control (invasibility) table."""
    print("\n## 2.2.4 Control: Profit vs ZIC Baseline (Profit Ratio)\n")
    print("| Strategy | BASE | BBBS | BSSS | EQL | RAN | PER | SHRT | TOK | SML | LAD | Mean |")
    print("|----------|------|------|------|-----|-----|-----|------|-----|-----|-----|------|")

    for trader in sorted(results.keys()):
        row = [trader]
        values = []
        for env in ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]:
            if env in results[trader]:
                val = results[trader][env]["profit_ratio"]
                row.append(f"{val:.2f}")
                values.append(val)
            else:
                row.append("-")

        mean = np.mean(values) if values else 0
        row.append(f"{mean:.2f}")
        print("| " + " | ".join(row) + " |")


def print_profit_dispersion_table(results: dict):
    """Print profit dispersion table."""
    print("\n#### 2.2.4 Profit Dispersion (RMS)")
    print("\n*Lower is better. Measures how evenly profits are distributed across traders.*\n")
    print("| Strategy | BASE | BBBS | BSSS | EQL | RAN | PER | SHRT | TOK | SML | LAD |")
    print("|----------|------|------|------|-----|-----|-----|------|-----|-----|-----|")

    for trader in sorted(results.keys()):
        row = [trader]
        for env in ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]:
            if env in results[trader]:
                val = results[trader][env]["profit_dispersion"]
                if np.isinf(val) or np.isnan(val):
                    row.append("-")
                else:
                    row.append(f"{val:.0f}")
            else:
                row.append("-")
        print("| " + " | ".join(row) + " |")


def print_rmsd_table(results: dict):
    """Print RMSD table."""
    print("\n#### 2.2.5 RMSD (Root Mean Squared Deviation)")
    print(
        "\n*Lower is better. RMSD = sqrt(mean((p_t - P*)^2)) measures price deviation from equilibrium.*\n"
    )
    print("| Strategy | BASE | BBBS | BSSS | EQL | RAN | PER | SHRT | TOK | SML | LAD |")
    print("|----------|------|------|------|-----|-----|-----|------|-----|-----|-----|")

    for trader in sorted(results.keys()):
        row = [trader]
        for env in ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]:
            if env in results[trader]:
                val = results[trader][env]["rmsd"]
                if np.isinf(val) or np.isnan(val):
                    row.append("-")
                else:
                    row.append(f"{val:.0f}")
            else:
                row.append("-")
        print("| " + " | ".join(row) + " |")


def print_trades_table(results: dict):
    """Print trades per period table."""
    print("\n#### 2.2.6 Trades per Period")
    print("\n*Average number of trades completed per period.*\n")
    print("| Strategy | BASE | BBBS | BSSS | EQL | RAN | PER | SHRT | TOK | SML | LAD |")
    print("|----------|------|------|------|-----|-----|-----|------|-----|-----|-----|")

    for trader in sorted(results.keys()):
        row = [trader]
        for env in ["BASE", "BBBS", "BSSS", "EQL", "RAN", "PER", "SHRT", "TOK", "SML", "LAD"]:
            if env in results[trader]:
                val = results[trader][env]["trades_per_period"]
                if np.isinf(val) or np.isnan(val):
                    row.append("-")
                else:
                    row.append(f"{val:.1f}")
            else:
                row.append("-")
        print("| " + " | ".join(row) + " |")


def main():
    print("=" * 60)
    print("Part 2: Santa Fe Tournament Results (12 Traders)")
    print("=" * 60)

    self_results = aggregate_self_play()
    ctrl_results = aggregate_control()

    print_self_play_table(self_results)
    print_vineff_table(self_results)
    print_volatility_table(self_results)
    print_profit_dispersion_table(self_results)
    print_rmsd_table(self_results)
    print_trades_table(self_results)
    print_control_table(ctrl_results)

    # Save to JSON
    output = {
        "self_play": self_results,
        "control": ctrl_results,
    }

    with open("results/p2_santafe_aggregated.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n\nResults saved to: results/p2_santafe_aggregated.json")


if __name__ == "__main__":
    main()
