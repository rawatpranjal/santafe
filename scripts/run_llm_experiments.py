#!/usr/bin/env python3
"""
LLM Experiments for Part 4.

Runs LLM agent experiments on BASE environment:
- 4.1 Against Control: 1 LLM buyer + 7 ZIC
- 4.3 Round Robin: 1 LLM + ZIC + ZIP + Skeleton + Kaplan
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load API key
api_key_path = Path(__file__).parent.parent / "key.txt"
if api_key_path.exists():
    os.environ["OPENAI_API_KEY"] = api_key_path.read_text().strip()
else:
    api_key_path = Path(__file__).parent.parent / "apikey.txt"
    if api_key_path.exists():
        os.environ["OPENAI_API_KEY"] = api_key_path.read_text().strip()

import numpy as np

from engine.agent_factory import create_agent
from engine.efficiency import (
    calculate_actual_surplus,
    calculate_max_surplus,
    extract_trades_from_orderbook,
    get_transaction_prices,
)
from engine.market import Market
from engine.token_generator import TokenGenerator


def run_tournament(
    buyer_types: list[str],
    seller_types: list[str],
    num_rounds: int = 5,
    num_periods: int = 10,
    num_steps: int = 100,
    num_tokens: int = 4,
    gametype: int = 6453,
    price_min: int = 1,
    price_max: int = 1000,
    seed: int = 42,
    llm_kwargs: dict = None,
) -> dict:
    """
    Run a tournament with specified agent types.

    Returns metrics: efficiency, volatility, profits by strategy.
    """
    if llm_kwargs is None:
        llm_kwargs = {}

    num_buyers = len(buyer_types)
    num_sellers = len(seller_types)

    # Results tracking
    efficiencies = []
    volatilities = []
    profits_by_type = {t: [] for t in set(buyer_types + seller_types)}
    trades_by_type = {t: [] for t in set(buyer_types + seller_types)}

    rng = np.random.default_rng(seed)

    for r in range(num_rounds):
        print(f"  Round {r+1}/{num_rounds}", end=" ", flush=True)

        # Create token generator for this round
        token_gen = TokenGenerator(gametype, num_tokens, seed + r * 1000)
        token_gen.new_round()

        # Create agents
        agents = []
        buyer_valuations = {}
        seller_costs = {}

        # Buyers
        for i, agent_type in enumerate(buyer_types):
            pid = i + 1
            tokens = token_gen.generate_tokens(True)
            buyer_valuations[pid] = tokens

            kwargs = {"num_buyers": num_buyers, "num_sellers": num_sellers}
            if agent_type.startswith("GPT"):
                kwargs.update(llm_kwargs)

            agent = create_agent(
                agent_type,
                pid,
                True,
                num_tokens,
                tokens,
                price_min=price_min,
                price_max=price_max,
                num_times=num_steps,
                seed=int(rng.integers(0, 100000)),
                **kwargs,
            )
            agents.append(agent)

        # Sellers
        for i, agent_type in enumerate(seller_types):
            pid = num_buyers + i + 1
            tokens = token_gen.generate_tokens(False)
            seller_costs[i + 1] = tokens  # Local seller ID

            kwargs = {"num_buyers": num_buyers, "num_sellers": num_sellers}
            if agent_type.startswith("GPT"):
                kwargs.update(llm_kwargs)

            agent = create_agent(
                agent_type,
                pid,
                False,
                num_tokens,
                tokens,
                price_min=price_min,
                price_max=price_max,
                num_times=num_steps,
                seed=int(rng.integers(0, 100000)),
                **kwargs,
            )
            agents.append(agent)

        # Calculate max surplus for efficiency
        buyer_vals_list = [buyer_valuations[a.player_id] for a in agents if a.is_buyer]
        seller_costs_list = [
            seller_costs[a.player_id - num_buyers] for a in agents if not a.is_buyer
        ]
        max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)

        # Run periods
        round_profits = {t: 0 for t in set(buyer_types + seller_types)}
        round_trades = {t: 0 for t in set(buyer_types + seller_types)}
        round_efficiencies = []
        round_volatilities = []

        for p in range(num_periods):
            # Create market
            market = Market(
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=price_min,
                price_max=price_max,
                num_times=num_steps,
                buyers=[a for a in agents if a.is_buyer],
                sellers=[a for a in agents if not a.is_buyer],
                seed=seed + r * 1000 + p,
            )

            # Start period
            for a in agents:
                a.start_period(p + 1)

            # Run market
            for t in range(num_steps):
                market.run_time_step()

            # End period
            for a in agents:
                a.end_period()

            # Extract trades using proper function
            trades = extract_trades_from_orderbook(market.orderbook, num_steps)

            # Calculate efficiency
            actual_surplus = calculate_actual_surplus(trades, buyer_valuations, seller_costs)
            efficiency = (actual_surplus / max_surplus * 100) if max_surplus > 0 else 0
            round_efficiencies.append(efficiency)

            # Calculate volatility from transaction prices
            prices = get_transaction_prices(market.orderbook, num_steps)
            if len(prices) > 1:
                volatility = (
                    float(np.std(prices) / np.mean(prices) * 100) if np.mean(prices) > 0 else 0
                )
            else:
                volatility = 0
            round_volatilities.append(volatility)

            # Collect profits
            for i, agent in enumerate(agents):
                if agent.is_buyer:
                    atype = buyer_types[i]
                else:
                    atype = seller_types[i - num_buyers]
                round_profits[atype] += agent.period_profit
                round_trades[atype] += agent.num_trades

        # Aggregate round results
        efficiencies.append(np.mean(round_efficiencies))
        volatilities.append(np.mean(round_volatilities))
        for t in round_profits:
            profits_by_type[t].append(round_profits[t])
            trades_by_type[t].append(round_trades[t])

        print(f"eff={np.mean(round_efficiencies):.1f}%")

    # Summarize
    results = {
        "efficiency_mean": float(np.mean(efficiencies)),
        "efficiency_std": float(np.std(efficiencies)),
        "volatility_mean": float(np.mean(volatilities)),
        "volatility_std": float(np.std(volatilities)),
        "profits": {
            t: {"mean": float(np.mean(p)), "std": float(np.std(p)), "total": float(np.sum(p))}
            for t, p in profits_by_type.items()
        },
        "trades": {
            t: {"mean": float(np.mean(tr)), "total": int(np.sum(tr))}
            for t, tr in trades_by_type.items()
        },
    }

    return results


def run_against_control():
    """
    4.1 Against Control: 1 LLM buyer + 7 ZIC agents.
    """
    print("\n" + "=" * 60)
    print("4.1 AGAINST CONTROL: 1 LLM (GPT-4o-mini) + 7 ZIC")
    print("=" * 60)

    # 1 LLM buyer + 3 ZIC buyers, 4 ZIC sellers
    buyer_types = ["GPT4-mini", "ZIC", "ZIC", "ZIC"]
    seller_types = ["ZIC", "ZIC", "ZIC", "ZIC"]

    results = run_tournament(
        buyer_types=buyer_types,
        seller_types=seller_types,
        num_rounds=5,
        num_periods=10,
        num_steps=100,
        llm_kwargs={"prompt_style": "deep"},
    )

    print("\n--- Results ---")
    print(f"Efficiency: {results['efficiency_mean']:.1f}% ± {results['efficiency_std']:.1f}%")
    print(f"Volatility: {results['volatility_mean']:.1f}% ± {results['volatility_std']:.1f}%")

    print("\nProfits:")
    for t, p in results["profits"].items():
        print(f"  {t}: {p['mean']:.1f} ± {p['std']:.1f} (total: {p['total']:.0f})")

    # Calculate profit ratio
    llm_profit = results["profits"].get("GPT4-mini", {}).get("total", 0)
    zic_profit = results["profits"].get("ZIC", {}).get("total", 0) / 7  # Per ZIC agent
    if zic_profit > 0:
        ratio = llm_profit / zic_profit
        print(f"\nLLM Profit Ratio vs ZIC: {ratio:.2f}x")
        results["profit_ratio_vs_zic"] = ratio

    return results


def run_round_robin():
    """
    4.3 Round Robin: 1 LLM + mixed strategies.
    """
    print("\n" + "=" * 60)
    print("4.3 ROUND ROBIN: LLM in Mixed Market")
    print("=" * 60)

    # 1 LLM buyer + ZIC + ZIP + Skeleton buyers, similar sellers
    buyer_types = ["GPT4-mini", "ZIC", "ZIP", "Skeleton"]
    seller_types = ["Kaplan", "ZIC", "ZIP", "Skeleton"]

    results = run_tournament(
        buyer_types=buyer_types,
        seller_types=seller_types,
        num_rounds=5,
        num_periods=10,
        num_steps=100,
        llm_kwargs={"prompt_style": "deep"},
    )

    print("\n--- Results ---")
    print(f"Efficiency: {results['efficiency_mean']:.1f}% ± {results['efficiency_std']:.1f}%")
    print(f"Volatility: {results['volatility_mean']:.1f}% ± {results['volatility_std']:.1f}%")

    print("\nProfits (total over 5 rounds × 10 periods):")
    profits_sorted = sorted(results["profits"].items(), key=lambda x: -x[1]["total"])
    for rank, (t, p) in enumerate(profits_sorted, 1):
        print(f"  {rank}. {t}: {p['total']:.0f}")

    # Calculate LLM rank
    llm_rank = next(i for i, (t, _) in enumerate(profits_sorted, 1) if t == "GPT4-mini")
    results["llm_rank"] = llm_rank
    print(f"\nLLM Rank: {llm_rank}/5")

    return results


def main():
    """Run all LLM experiments."""
    print("=" * 60)
    print("PART 4: LLM EXPERIMENTS (BASE Environment Only)")
    print("=" * 60)
    print("Model: GPT-4o-mini with deep context prompts")
    print("Config: 5 rounds × 10 periods × 100 steps")
    print("=" * 60)

    results = {}

    # 4.1 Against Control
    results["against_control"] = run_against_control()

    # 4.3 Round Robin
    results["round_robin"] = run_round_robin()

    # Save results
    output_path = Path("results/p4_llm_experiments.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "timestamp": datetime.now().isoformat(),
        "model": "gpt-4o-mini",
        "prompt_style": "deep",
        "environment": "BASE",
        "config": {
            "num_rounds": 5,
            "num_periods": 10,
            "num_steps": 100,
            "num_tokens": 4,
            "gametype": 6453,
        },
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_path}")

    # Summary
    print("\n--- SUMMARY ---")
    ctrl = results["against_control"]
    rr = results["round_robin"]
    print("4.1 Against Control:")
    print(f"  Efficiency: {ctrl['efficiency_mean']:.1f}%")
    print(f"  LLM Profit Ratio: {ctrl.get('profit_ratio_vs_zic', 'N/A'):.2f}x vs ZIC")
    print("4.3 Round Robin:")
    print(f"  Efficiency: {rr['efficiency_mean']:.1f}%")
    print(f"  LLM Rank: {rr['llm_rank']}/5")


if __name__ == "__main__":
    main()
