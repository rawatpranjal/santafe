#!/usr/bin/env python3
"""
Individual Profit Analysis Script.

Runs pairwise tournaments and generates detailed per-agent profit breakdowns:
1. Per-agent profit in mixed markets
2. Profit deviation from equilibrium
3. Buyer vs Seller profit split
"""

import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from engine.tournament import Tournament


def run_pairwise_tournament(
    type_a: str,
    type_b: str,
    num_rounds: int = 50,
    num_periods: int = 10,
    seed: int = 42
) -> pd.DataFrame:
    """Run a pairwise tournament with 4 of each type per side."""

    config = OmegaConf.create({
        "experiment": {
            "name": f"{type_a}_vs_{type_b}_profit_analysis",
            "num_rounds": num_rounds,
            "rng_seed_values": seed,
            "rng_seed_auction": seed + 1000,
        },
        "market": {
            "gametype": 6453,
            "num_tokens": 4,
            "min_price": 1,
            "max_price": 1000,
            "num_periods": num_periods,
            "num_steps": 100,
            "token_mode": "santafe",
        },
        "agents": {
            # 4 of each type per side
            "buyer_types": [type_a, type_a, type_b, type_b],
            "seller_types": [type_a, type_a, type_b, type_b],
        }
    })

    tournament = Tournament(config)
    return tournament.run()


def analyze_profits(df: pd.DataFrame, type_a: str, type_b: str) -> dict:
    """Analyze individual profits from tournament results."""

    results = {
        "per_agent": [],
        "by_type_role": {},
        "summary": {}
    }

    # Per-agent aggregation across all rounds/periods
    agent_stats = df.groupby(["agent_id", "agent_type", "is_buyer"]).agg({
        "period_profit": "sum",
        "agent_eq_profit": "sum",
        "profit_deviation": "sum",
        "num_trades": "sum"
    }).reset_index()

    # Calculate deviation percentage
    agent_stats["deviation_pct"] = np.where(
        agent_stats["agent_eq_profit"] != 0,
        (agent_stats["profit_deviation"] / agent_stats["agent_eq_profit"]) * 100,
        0
    )

    # Store per-agent results
    for _, row in agent_stats.iterrows():
        results["per_agent"].append({
            "agent_id": int(row["agent_id"]),
            "type": row["agent_type"],
            "role": "Buyer" if row["is_buyer"] else "Seller",
            "total_profit": int(row["period_profit"]),
            "eq_profit": int(row["agent_eq_profit"]),
            "deviation": int(row["profit_deviation"]),
            "deviation_pct": round(row["deviation_pct"], 1),
            "num_trades": int(row["num_trades"])
        })

    # By type and role
    type_role_stats = df.groupby(["agent_type", "is_buyer"]).agg({
        "period_profit": "sum",
        "agent_eq_profit": "sum",
        "profit_deviation": "sum",
        "num_trades": "sum"
    }).reset_index()

    for _, row in type_role_stats.iterrows():
        key = f"{row['agent_type']}_{'Buyer' if row['is_buyer'] else 'Seller'}"
        results["by_type_role"][key] = {
            "total_profit": int(row["period_profit"]),
            "eq_profit": int(row["agent_eq_profit"]),
            "deviation": int(row["profit_deviation"]),
            "num_trades": int(row["num_trades"])
        }

    # Summary by type (all roles combined)
    type_stats = df.groupby("agent_type").agg({
        "period_profit": "sum",
        "agent_eq_profit": "sum",
        "profit_deviation": "sum",
        "num_trades": "sum"
    }).reset_index()

    for _, row in type_stats.iterrows():
        results["summary"][row["agent_type"]] = {
            "total_profit": int(row["period_profit"]),
            "eq_profit": int(row["agent_eq_profit"]),
            "deviation": int(row["profit_deviation"]),
            "num_trades": int(row["num_trades"])
        }

    return results


def print_report(results: dict, type_a: str, type_b: str, num_rounds: int, num_periods: int):
    """Print formatted profit analysis report."""

    print(f"\n{'='*70}")
    print(f"INDIVIDUAL PROFIT ANALYSIS: {type_a} vs {type_b}")
    print(f"({num_rounds} rounds × {num_periods} periods)")
    print(f"{'='*70}\n")

    # Per-agent breakdown
    print("### Per-Agent Profit Breakdown\n")
    print(f"| Agent | Type | Role   | Total Profit | Eq Profit | Deviation | Dev % | Trades |")
    print(f"|-------|------|--------|--------------|-----------|-----------|-------|--------|")

    for agent in sorted(results["per_agent"], key=lambda x: x["agent_id"]):
        sign = "+" if agent["deviation"] >= 0 else ""
        print(f"| {agent['agent_id']:5} | {agent['type']:4} | {agent['role']:6} | "
              f"{agent['total_profit']:+12,} | {agent['eq_profit']:+9,} | "
              f"{sign}{agent['deviation']:+8,} | {agent['deviation_pct']:+5.1f}% | {agent['num_trades']:6} |")

    print()

    # By type and role
    print("### Profit by Type and Role\n")
    print(f"| Type | Role   | Total Profit | Eq Profit | Deviation | Trades |")
    print(f"|------|--------|--------------|-----------|-----------|--------|")

    for key in sorted(results["by_type_role"].keys()):
        stats = results["by_type_role"][key]
        type_name, role = key.rsplit("_", 1)
        print(f"| {type_name:4} | {role:6} | {stats['total_profit']:+12,} | "
              f"{stats['eq_profit']:+9,} | {stats['deviation']:+9,} | {stats['num_trades']:6} |")

    print()

    # Summary
    print("### Summary by Strategy Type\n")
    print(f"| Type | Total Profit | Eq Profit | Deviation | Advantage |")
    print(f"|------|--------------|-----------|-----------|-----------|")

    profits = {t: s["total_profit"] for t, s in results["summary"].items()}
    max_profit = max(profits.values())

    for type_name, stats in sorted(results["summary"].items()):
        adv = "WINNER" if stats["total_profit"] == max_profit else ""
        print(f"| {type_name:4} | {stats['total_profit']:+12,} | "
              f"{stats['eq_profit']:+9,} | {stats['deviation']:+9,} | {adv:9} |")

    print()

    # Buyer vs Seller analysis
    print("### Buyer vs Seller Split\n")

    buyer_profit = sum(
        s["total_profit"] for k, s in results["by_type_role"].items() if "Buyer" in k
    )
    seller_profit = sum(
        s["total_profit"] for k, s in results["by_type_role"].items() if "Seller" in k
    )
    total_profit = buyer_profit + seller_profit

    buyer_pct = (buyer_profit / total_profit * 100) if total_profit else 0
    seller_pct = (seller_profit / total_profit * 100) if total_profit else 0

    print(f"| Side    | Total Profit | Share % |")
    print(f"|---------|--------------|---------|")
    print(f"| Buyers  | {buyer_profit:+12,} | {buyer_pct:6.1f}% |")
    print(f"| Sellers | {seller_profit:+12,} | {seller_pct:6.1f}% |")
    print(f"| **Total** | {total_profit:+12,} | 100.0%  |")

    print()

    # Type advantage breakdown
    print("### Strategy Advantage by Role\n")

    for type_name in results["summary"].keys():
        buyer_key = f"{type_name}_Buyer"
        seller_key = f"{type_name}_Seller"

        b_profit = results["by_type_role"].get(buyer_key, {}).get("total_profit", 0)
        s_profit = results["by_type_role"].get(seller_key, {}).get("total_profit", 0)

        print(f"**{type_name}**: Buyers {b_profit:+,} | Sellers {s_profit:+,} | Total {b_profit + s_profit:+,}")

    print()


def generate_markdown_section(results: dict, type_a: str, type_b: str, num_rounds: int, num_periods: int) -> str:
    """Generate markdown for results.md."""

    lines = []
    lines.append(f"### 2.6 Individual Profit Analysis ({type_a} vs {type_b})\n")
    lines.append(f"**Config**: 4 {type_a} + 4 {type_b} per side, {num_rounds} rounds, {num_periods} periods\n")

    # Per-agent table
    lines.append("#### Per-Agent Breakdown\n")
    lines.append("| Agent | Type | Role | Total Profit | Eq Profit | Deviation | Dev % |")
    lines.append("|-------|------|------|--------------|-----------|-----------|-------|")

    for agent in sorted(results["per_agent"], key=lambda x: x["agent_id"]):
        sign = "+" if agent["deviation"] >= 0 else ""
        lines.append(f"| {agent['agent_id']} | {agent['type']} | {agent['role']} | "
                    f"{agent['total_profit']:+,} | {agent['eq_profit']:+,} | "
                    f"{sign}{agent['deviation']:,} | {agent['deviation_pct']:+.1f}% |")

    lines.append("")

    # Summary table
    lines.append("#### Strategy Summary\n")
    lines.append("| Type | Buyers | Sellers | Total | Advantage |")
    lines.append("|------|--------|---------|-------|-----------|")

    profits = {t: s["total_profit"] for t, s in results["summary"].items()}
    max_profit = max(profits.values())

    for type_name in sorted(results["summary"].keys()):
        buyer_key = f"{type_name}_Buyer"
        seller_key = f"{type_name}_Seller"

        b_profit = results["by_type_role"].get(buyer_key, {}).get("total_profit", 0)
        s_profit = results["by_type_role"].get(seller_key, {}).get("total_profit", 0)
        total = results["summary"][type_name]["total_profit"]

        adv = "**WINNER**" if total == max_profit else ""
        lines.append(f"| {type_name} | {b_profit:+,} | {s_profit:+,} | {total:+,} | {adv} |")

    lines.append("")

    # Buyer vs Seller
    buyer_profit = sum(
        s["total_profit"] for k, s in results["by_type_role"].items() if "Buyer" in k
    )
    seller_profit = sum(
        s["total_profit"] for k, s in results["by_type_role"].items() if "Seller" in k
    )
    total_profit = buyer_profit + seller_profit

    buyer_pct = (buyer_profit / total_profit * 100) if total_profit else 0
    seller_pct = (seller_profit / total_profit * 100) if total_profit else 0

    lines.append("#### Buyer vs Seller Split\n")
    lines.append("| Side | Profit | Share |")
    lines.append("|------|--------|-------|")
    lines.append(f"| Buyers | {buyer_profit:+,} | {buyer_pct:.1f}% |")
    lines.append(f"| Sellers | {seller_profit:+,} | {seller_pct:.1f}% |")
    lines.append("")

    # Key findings
    lines.append("#### Key Findings\n")

    # Calculate which type wins on each side
    type_a_buyer = results["by_type_role"].get(f"{type_a}_Buyer", {}).get("total_profit", 0)
    type_b_buyer = results["by_type_role"].get(f"{type_b}_Buyer", {}).get("total_profit", 0)
    type_a_seller = results["by_type_role"].get(f"{type_a}_Seller", {}).get("total_profit", 0)
    type_b_seller = results["by_type_role"].get(f"{type_b}_Seller", {}).get("total_profit", 0)

    buyer_winner = type_a if type_a_buyer > type_b_buyer else type_b
    seller_winner = type_a if type_a_seller > type_b_seller else type_b
    overall_winner = type_a if results["summary"][type_a]["total_profit"] > results["summary"][type_b]["total_profit"] else type_b

    lines.append(f"1. **Overall Winner**: {overall_winner}")
    lines.append(f"2. **Buyer Side**: {buyer_winner} earns more as buyer")
    lines.append(f"3. **Seller Side**: {seller_winner} earns more as seller")
    lines.append(f"4. **Market Balance**: {'Buyers' if buyer_pct > 50 else 'Sellers'} capture {max(buyer_pct, seller_pct):.1f}% of total surplus")
    lines.append("")
    lines.append(f"- **Date**: 2025-11-28\n")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import logging

    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description="Individual Profit Analysis")
    parser.add_argument("--type-a", default="ZIP", help="First strategy type")
    parser.add_argument("--type-b", default="ZIC", help="Second strategy type")
    parser.add_argument("--rounds", type=int, default=50, help="Number of rounds")
    parser.add_argument("--periods", type=int, default=10, help="Periods per round")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--markdown", action="store_true", help="Output markdown for results.md")

    args = parser.parse_args()

    print(f"Running {args.type_a} vs {args.type_b} tournament...")
    df = run_pairwise_tournament(
        args.type_a, args.type_b,
        num_rounds=args.rounds,
        num_periods=args.periods,
        seed=args.seed
    )

    print(f"Analyzing profits...")
    results = analyze_profits(df, args.type_a, args.type_b)

    if args.markdown:
        md = generate_markdown_section(results, args.type_a, args.type_b, args.rounds, args.periods)
        print(md)
    else:
        print_report(results, args.type_a, args.type_b, args.rounds, args.periods)
