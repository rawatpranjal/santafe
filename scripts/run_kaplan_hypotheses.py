#!/usr/bin/env python3
"""
Run 8 targeted hypothesis tests for Kaplan win/loss conditions.

Hypotheses:
  H1: Kaplan wins when noise traders (ZIC) > 50% of opponents
  H2: Kaplan loses when competing with other strategic waiters
  H3: Kaplan advantage is HIGHEST in single-token environments
  H4: Time pressure crashes Kaplan vs Kaplan, not vs ZIC
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import argparse

import pandas as pd

from engine.agent_factory import create_agent
from engine.efficiency import (
    calculate_actual_surplus,
    calculate_allocative_efficiency,
    calculate_max_surplus,
    extract_trades_from_orderbook,
)
from engine.market import Market
from engine.token_generator import TokenGenerator

# Environment parameters
ENV_PARAMS = {
    "base": {"num_tokens": 4, "num_steps": 100, "gametype": 6453},
    "tok": {"num_tokens": 1, "num_steps": 100, "gametype": 6453},
    "shrt": {"num_tokens": 4, "num_steps": 20, "gametype": 6453},
}

# The 8 experiments
EXPERIMENTS = [
    # H1: Noise Trader Threshold
    {
        "id": "H1.1",
        "hypothesis": "ZIC > 50%",
        "buyers": ["Kaplan", "ZIC", "ZIC", "ZIP"],
        "sellers": ["ZIC", "ZIC", "ZIP", "ZIP"],
        "env": "base",
        "description": "1 Kaplan + 4 ZIC + 3 ZIP (57% ZIC)",
    },
    {
        "id": "H1.2",
        "hypothesis": "ZIC < 50%",
        "buyers": ["Kaplan", "ZIC", "ZIP", "ZIP"],
        "sellers": ["ZIP", "ZIP", "ZIP", "ZIP"],
        "env": "base",
        "description": "1 Kaplan + 1 ZIC + 6 ZIP (14% ZIC)",
    },
    # H2: Sniper Competition
    {
        "id": "H2.1",
        "hypothesis": "Sniper deadlock",
        "buyers": ["Kaplan", "Kaplan"],
        "sellers": ["Kaplan", "Kaplan"],
        "env": "base",
        "description": "2v2 Kaplan vs Kaplan",
    },
    {
        "id": "H2.2",
        "hypothesis": "Sniper competition",
        "buyers": ["Kaplan", "Skeleton", "ZIC", "ZIC"],
        "sellers": ["Kaplan", "Skeleton", "ZIC", "ZIC"],
        "env": "base",
        "description": "Kaplan + Skeleton + 2 ZIC each side",
    },
    # H3: Token Count
    {
        "id": "H3.1",
        "hypothesis": "Few tokens",
        "buyers": ["Kaplan", "ZIC", "ZIC", "ZIC"],
        "sellers": ["ZIC", "ZIC", "ZIC", "ZIC"],
        "env": "base",
        "tokens_override": 2,
        "description": "1 Kaplan vs 7 ZIC (2 tokens)",
    },
    {
        "id": "H3.2",
        "hypothesis": "Single token mixed",
        "buyers": ["Kaplan", "Skeleton", "ZIC", "ZIP"],
        "sellers": ["Kaplan", "Skeleton", "ZIC", "ZIP"],
        "env": "tok",
        "description": "4v4 Mixed (TOK env, 1 token)",
    },
    # H4: Time Pressure
    {
        "id": "H4.1",
        "hypothesis": "Time + self-play",
        "buyers": ["Kaplan", "Kaplan", "Kaplan", "Kaplan"],
        "sellers": ["Kaplan", "Kaplan", "Kaplan", "Kaplan"],
        "env": "shrt",
        "description": "8x Kaplan self-play (SHRT, 20 steps)",
    },
    {
        "id": "H4.2",
        "hypothesis": "Time + mixed",
        "buyers": ["Kaplan", "Skeleton", "ZIC", "ZIP"],
        "sellers": ["Kaplan", "Skeleton", "ZIC", "ZIP"],
        "env": "shrt",
        "description": "4v4 Mixed (SHRT, 20 steps)",
    },
]


def run_experiment(
    exp: dict,
    num_rounds: int = 50,
    num_periods: int = 10,
    price_min: int = 1,
    price_max: int = 1000,
    seed: int = 123,  # Match tournament's rng_seed_values
) -> pd.DataFrame:
    """Run a single hypothesis experiment.

    NOTE: Uses single TokenGenerator with sequential new_round() calls
    to match tournament.py behavior exactly.
    """
    env = ENV_PARAMS[exp["env"]]
    num_tokens = exp.get("tokens_override", env["num_tokens"])
    num_steps = env["num_steps"]

    buyer_types = exp["buyers"]
    seller_types = exp["sellers"]
    num_buyers = len(buyer_types)
    num_sellers = len(seller_types)

    all_results = []

    # FIXED: Single TokenGenerator, reused across rounds (matches tournament.py)
    gametype = env["gametype"]
    token_gen = TokenGenerator(gametype, num_tokens, seed)
    rng_seed_auction = 42  # Match tournament's rng_seed_auction

    for r in range(num_rounds):
        # Advance round state (matches tournament.py line 128)
        token_gen.new_round()

        # Create agents with consistent seeds (matches tournament.py)
        buyers = []
        for i, agent_type in enumerate(buyer_types):
            player_id = i + 1
            vals = token_gen.generate_tokens(is_buyer=True)
            agent = create_agent(
                agent_type,
                player_id=player_id,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=vals,
                price_min=price_min,
                price_max=price_max,
                num_times=num_steps,
                seed=rng_seed_auction + player_id,
            )
            agent.start_round(vals)
            buyers.append(agent)

        sellers = []
        for i, agent_type in enumerate(seller_types):
            player_id = num_buyers + i + 1
            costs = token_gen.generate_tokens(is_buyer=False)
            agent = create_agent(
                agent_type,
                player_id=player_id,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=costs,
                price_min=price_min,
                price_max=price_max,
                num_times=num_steps,
                seed=rng_seed_auction + player_id,
            )
            agent.start_round(costs)
            sellers.append(agent)

        round_results = []

        for p in range(num_periods):
            market = Market(
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                num_times=num_steps,
                price_min=price_min,
                price_max=price_max,
                buyers=buyers,
                sellers=sellers,
            )
            market.set_period(r + 1, p + 1)

            for agent in buyers + sellers:
                agent.start_period(p + 1)

            while market.current_time < market.num_times:
                market.run_time_step()

            # Calculate efficiency
            trades = extract_trades_from_orderbook(market.orderbook, num_steps)

            buyer_vals_list = [list(b.valuations) for b in buyers]
            seller_costs_list = [list(s.valuations) for s in sellers]

            # Orderbook uses 1-indexed buyer_id (1..num_buyers) and seller_id (1..num_sellers)
            buyer_vals_dict = {i + 1: list(buyers[i].valuations) for i in range(len(buyers))}
            seller_costs_dict = {i + 1: list(sellers[i].valuations) for i in range(len(sellers))}

            max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
            actual_surplus = calculate_actual_surplus(trades, buyer_vals_dict, seller_costs_dict)
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

            # Record results for each agent
            for agent in buyers + sellers:
                round_results.append(
                    {
                        "round": r + 1,
                        "period": p + 1,
                        "agent_id": agent.player_id,
                        "agent_type": agent.__class__.__name__,
                        "is_buyer": agent.is_buyer,
                        "period_profit": agent.period_profit,
                        "num_trades": agent.num_trades,
                        "efficiency": efficiency,
                    }
                )

            for agent in buyers + sellers:
                agent.end_period()

        all_results.extend(round_results)

    return pd.DataFrame(all_results)


def analyze_results(df: pd.DataFrame, exp: dict) -> dict:
    """Analyze results to compute Kaplan metrics."""

    # Aggregate by agent type
    by_type = (
        df.groupby("agent_type")
        .agg(
            {
                "period_profit": "sum",
                "num_trades": "sum",
                "efficiency": "mean",
            }
        )
        .reset_index()
    )

    # Get Kaplan profit
    kaplan_row = by_type[by_type["agent_type"] == "Kaplan"]
    if kaplan_row.empty:
        kaplan_profit = 0
        kaplan_trades = 0
    else:
        kaplan_profit = kaplan_row["period_profit"].values[0]
        kaplan_trades = kaplan_row["num_trades"].values[0]

    # Get opponent profit (non-Kaplan)
    opponent_rows = by_type[by_type["agent_type"] != "Kaplan"]
    if opponent_rows.empty:
        opponent_profit = kaplan_profit  # self-play
        opponent_count = len(exp["buyers"]) + len(exp["sellers"]) - 1
    else:
        opponent_profit = opponent_rows["period_profit"].sum()
        opponent_count = len(opponent_rows)

    # Calculate ratio
    kaplan_count = exp["buyers"].count("Kaplan") + exp["sellers"].count("Kaplan")
    total_agents = len(exp["buyers"]) + len(exp["sellers"])

    # Per-agent averages
    kaplan_per_agent = kaplan_profit / kaplan_count if kaplan_count > 0 else 0
    opponent_per_agent = (
        opponent_profit / (total_agents - kaplan_count) if (total_agents - kaplan_count) > 0 else 0
    )

    # Profit ratio
    if opponent_per_agent > 0:
        profit_ratio = kaplan_per_agent / opponent_per_agent
    elif kaplan_per_agent > 0:
        profit_ratio = float("inf")
    else:
        profit_ratio = 1.0

    # Rank Kaplan among all strategies
    by_type["per_agent_profit"] = by_type["period_profit"] / by_type["agent_type"].map(
        lambda t: exp["buyers"].count(t) + exp["sellers"].count(t)
    )
    by_type_sorted = by_type.sort_values("per_agent_profit", ascending=False)
    types_ranked = by_type_sorted["agent_type"].tolist()
    kaplan_rank = (
        types_ranked.index("Kaplan") + 1 if "Kaplan" in types_ranked else len(types_ranked)
    )

    # Efficiency
    avg_efficiency = df["efficiency"].mean()

    # Result determination
    if exp["id"].startswith("H2.1") or exp["id"].startswith("H4.1"):
        # Self-play: check efficiency
        result = "DEADLOCK" if avg_efficiency < 80 else "OK"
    elif profit_ratio >= 1.0:
        result = "WIN"
    else:
        result = "LOSE"

    return {
        "test": exp["id"],
        "hypothesis": exp["hypothesis"],
        "config": exp["description"],
        "kaplan_profit": kaplan_per_agent,
        "opponent_profit": opponent_per_agent,
        "profit_ratio": profit_ratio,
        "kaplan_rank": f"{kaplan_rank}/{len(types_ranked)}",
        "efficiency": f"{avg_efficiency:.1f}%",
        "result": result,
    }


def main():
    parser = argparse.ArgumentParser(description="Kaplan Hypothesis Testing")
    parser.add_argument("--num_rounds", type=int, default=50, help="Rounds per experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("KAPLAN WIN/LOSS HYPOTHESIS TESTING")
    print("=" * 70)
    print(f"Experiments: {len(EXPERIMENTS)}")
    print(f"Rounds per experiment: {args.num_rounds}")
    print(f"Seed: {args.seed}")
    print()

    results = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"[{i}/{len(EXPERIMENTS)}] {exp['id']}: {exp['description']}...", end=" ", flush=True)

        df = run_experiment(exp, num_rounds=args.num_rounds, seed=args.seed)
        analysis = analyze_results(df, exp)
        results.append(analysis)

        # Save raw data
        output_dir = Path("results/kaplan_hypotheses")
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / f"{exp['id']}_raw.csv", index=False)

        print(
            f"ratio={analysis['profit_ratio']:.2f}x, eff={analysis['efficiency']}, {analysis['result']}"
        )

    # Create summary DataFrame
    summary_df = pd.DataFrame(results)

    # Print paper-ready table
    print("\n" + "=" * 70)
    print("PAPER-READY TABLE: Kaplan Win/Loss Conditions")
    print("=" * 70)
    print()
    print("| Test | Hypothesis | Configuration | Profit Ratio | Rank | Efficiency | Result |")
    print("|------|------------|---------------|--------------|------|------------|--------|")
    for _, row in summary_df.iterrows():
        ratio_str = f"{row['profit_ratio']:.2f}x" if row["profit_ratio"] < 100 else "N/A"
        print(
            f"| {row['test']} | {row['hypothesis']} | {row['config'][:30]}... | {ratio_str} | {row['kaplan_rank']} | {row['efficiency']} | **{row['result']}** |"
        )

    # Save summary
    summary_df.to_csv(output_dir / "summary.csv", index=False)

    print("\n" + "=" * 70)
    print("HYPOTHESIS VERDICTS")
    print("=" * 70)

    # H1: ZIC Threshold
    h1_1 = summary_df[summary_df["test"] == "H1.1"].iloc[0]
    h1_2 = summary_df[summary_df["test"] == "H1.2"].iloc[0]
    h1_verdict = (
        "SUPPORTED"
        if h1_1["profit_ratio"] > 1.0 and h1_2["profit_ratio"] < 1.0
        else "NOT SUPPORTED"
    )
    print(f"\nH1 (ZIC > 50% = WIN): {h1_verdict}")
    print(f"  - H1.1 (57% ZIC): ratio={h1_1['profit_ratio']:.2f}x → {h1_1['result']}")
    print(f"  - H1.2 (14% ZIC): ratio={h1_2['profit_ratio']:.2f}x → {h1_2['result']}")

    # H2: Sniper Competition
    h2_1 = summary_df[summary_df["test"] == "H2.1"].iloc[0]
    h2_2 = summary_df[summary_df["test"] == "H2.2"].iloc[0]
    h2_1_eff = float(h2_1["efficiency"].replace("%", ""))
    h2_verdict = "SUPPORTED" if h2_1_eff < 80 else "NOT SUPPORTED"
    print(f"\nH2 (Sniper competition = DEADLOCK): {h2_verdict}")
    print(f"  - H2.1 (Kaplan vs Kaplan): eff={h2_1['efficiency']} → {h2_1['result']}")
    print(f"  - H2.2 (Kaplan + Skeleton): rank={h2_2['kaplan_rank']} → {h2_2['result']}")

    # H3: Token Count
    h3_1 = summary_df[summary_df["test"] == "H3.1"].iloc[0]
    h3_2 = summary_df[summary_df["test"] == "H3.2"].iloc[0]
    h3_verdict = "SUPPORTED" if h3_2["result"] == "WIN" else "NOT SUPPORTED"
    print(f"\nH3 (Single token = WIN even in mixed): {h3_verdict}")
    print(f"  - H3.1 (2 tokens vs ZIC): ratio={h3_1['profit_ratio']:.2f}x → {h3_1['result']}")
    print(f"  - H3.2 (1 token mixed): ratio={h3_2['profit_ratio']:.2f}x → {h3_2['result']}")

    # H4: Time Pressure
    h4_1 = summary_df[summary_df["test"] == "H4.1"].iloc[0]
    h4_2 = summary_df[summary_df["test"] == "H4.2"].iloc[0]
    h4_1_eff = float(h4_1["efficiency"].replace("%", ""))
    h4_verdict = "SUPPORTED" if h4_1_eff < 80 else "NOT SUPPORTED"
    print(f"\nH4 (Time pressure crashes self-play): {h4_verdict}")
    print(f"  - H4.1 (SHRT self-play): eff={h4_1['efficiency']} → {h4_1['result']}")
    print(f"  - H4.2 (SHRT mixed): rank={h4_2['kaplan_rank']} → {h4_2['result']}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
