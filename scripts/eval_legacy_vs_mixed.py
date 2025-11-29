"""
Evaluate legacy agents (Kaplan, GD, Lin, ZIC) against Mixed opponents.

This provides the baseline comparison points for the learning curve.
All agents should be evaluated in the same 1v7 format used for PPO.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from engine.market import Market
from engine.token_generator import TokenGenerator
from engine.agent_factory import create_agent
from engine.metrics import calculate_equilibrium_profit


def run_episode(
    test_agent_type: str,
    opponent_pool: list[str],
    num_buyers: int = 4,
    num_sellers: int = 4,
    num_tokens: int = 4,
    num_times: int = 100,
    price_min: int = 1,
    price_max: int = 100,
    seed: int = 42,
    test_is_buyer: bool = True
) -> dict:
    """
    Run a single episode with test agent vs mixed opponents.

    Args:
        test_agent_type: Type of agent to test (e.g., "Kaplan", "GD")
        opponent_pool: List of opponent types to sample from
        num_buyers/sellers: Market composition
        num_tokens: Tokens per agent
        num_times: Time steps per episode
        price_min/max: Price range
        seed: Random seed
        test_is_buyer: Whether test agent is a buyer

    Returns:
        Dict with episode metrics
    """
    rng = np.random.default_rng(seed)
    token_gen = TokenGenerator(1111, num_tokens, seed)

    buyers = []
    sellers = []
    test_agent = None

    # Create buyers
    for i in range(num_buyers):
        pid = i + 1
        token_gen.new_round()
        tokens = token_gen.generate_tokens(True)

        if test_is_buyer and i == 0:
            # First buyer is our test agent
            agent = create_agent(
                test_agent_type, pid, True, num_tokens, tokens,
                price_min=price_min, price_max=price_max,
                num_times=num_times, seed=rng.integers(0, 100000)
            )
            test_agent = agent
        else:
            # Sample opponent from pool
            opp_type = rng.choice(opponent_pool)
            agent = create_agent(
                opp_type, pid, True, num_tokens, tokens,
                price_min=price_min, price_max=price_max,
                num_times=num_times, seed=rng.integers(0, 100000)
            )
        buyers.append(agent)

    # Create sellers
    for i in range(num_sellers):
        pid = num_buyers + i + 1
        token_gen.new_round()
        tokens = token_gen.generate_tokens(False)

        if not test_is_buyer and i == 0:
            # First seller is our test agent
            agent = create_agent(
                test_agent_type, pid, False, num_tokens, tokens,
                price_min=price_min, price_max=price_max,
                num_times=num_times, seed=rng.integers(0, 100000)
            )
            test_agent = agent
        else:
            # Sample opponent from pool
            opp_type = rng.choice(opponent_pool)
            agent = create_agent(
                opp_type, pid, False, num_tokens, tokens,
                price_min=price_min, price_max=price_max,
                num_times=num_times, seed=rng.integers(0, 100000)
            )
        sellers.append(agent)

    # Create market
    market = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        num_times=num_times,
        price_min=price_min,
        price_max=price_max,
        buyers=buyers,
        sellers=sellers,
        seed=seed
    )

    # Start all agents
    for a in buyers + sellers:
        a.start_period(1)

    # Run market
    for t in range(num_times):
        market.run_time_step()

    # Calculate metrics
    all_vals = [v for b in buyers for v in b.valuations]
    all_costs = [v for s in sellers for v in s.valuations]
    max_profit = calculate_equilibrium_profit(all_vals, all_costs)

    actual_profit = sum(a.period_profit for a in buyers + sellers)
    efficiency = actual_profit / max_profit if max_profit > 0 else 0

    test_profit = test_agent.period_profit if test_agent else 0
    test_trades = test_agent.num_trades if test_agent else 0

    return {
        "test_profit": test_profit,
        "test_trades": test_trades,
        "market_efficiency": efficiency,
        "total_profit": actual_profit,
        "max_profit": max_profit
    }


def evaluate_agent(
    agent_type: str,
    opponent_pool: list[str],
    num_episodes: int = 100,
    seed: int = 42
) -> dict:
    """
    Evaluate an agent type over multiple episodes.

    Returns:
        Dict with average metrics
    """
    profits = []
    trades = []
    efficiencies = []

    for ep in range(num_episodes):
        result = run_episode(
            test_agent_type=agent_type,
            opponent_pool=opponent_pool,
            seed=seed + ep,
            test_is_buyer=True
        )
        profits.append(result["test_profit"])
        trades.append(result["test_trades"])
        efficiencies.append(result["market_efficiency"])

    return {
        "avg_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits)),
        "avg_trades": float(np.mean(trades)),
        "avg_efficiency": float(np.mean(efficiencies)),
        "num_episodes": num_episodes
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate legacy agents vs Mixed")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes per agent")
    parser.add_argument("--output", type=str, default="./results/legacy_vs_mixed.json",
                       help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Agent types to evaluate
    agents_to_test = ["ZIC", "Kaplan", "GD", "Lin"]

    # Mixed opponent pool (strategies >= ZIC baseline)
    opponent_pool = ["ZIC", "Kaplan", "GD", "Lin"]

    print("=" * 60)
    print("LEGACY AGENTS VS MIXED EVALUATION")
    print("=" * 60)
    print(f"Episodes per agent: {args.episodes}")
    print(f"Opponent pool: {opponent_pool}")
    print("=" * 60)

    results = {}

    for agent_type in agents_to_test:
        print(f"\nEvaluating {agent_type}...")
        metrics = evaluate_agent(
            agent_type=agent_type,
            opponent_pool=opponent_pool,
            num_episodes=args.episodes,
            seed=args.seed
        )
        results[agent_type] = metrics
        print(f"  Avg profit: {metrics['avg_profit']:.2f} +/- {metrics['std_profit']:.2f}")
        print(f"  Avg trades: {metrics['avg_trades']:.1f}")
        print(f"  Avg efficiency: {metrics['avg_efficiency']:.2%}")

    # Calculate profit ratios relative to ZIC
    zic_profit = results["ZIC"]["avg_profit"]
    if zic_profit > 0:
        for agent_type in agents_to_test:
            ratio = results[agent_type]["avg_profit"] / zic_profit
            results[agent_type]["profit_ratio_vs_zic"] = ratio

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "episodes_per_agent": args.episodes,
        "opponent_pool": opponent_pool,
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Agent':<10} {'Profit':>10} {'Ratio':>10} {'Efficiency':>12}")
    print("-" * 45)
    for agent_type in agents_to_test:
        m = results[agent_type]
        ratio = m.get("profit_ratio_vs_zic", 1.0)
        print(f"{agent_type:<10} {m['avg_profit']:>10.2f} {ratio:>10.2f}x {m['avg_efficiency']:>11.2%}")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
