"""
Evaluate LLM agents against Mixed opponents.

This evaluates GPT-4o-mini in 1v7 format against mixed opponents
to benchmark zero-shot LLM performance.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load API key
api_key_path = Path(__file__).parent.parent / "apikey.txt"
if api_key_path.exists():
    os.environ["OPENAI_API_KEY"] = api_key_path.read_text().strip()

import numpy as np
from engine.market import Market
from engine.token_generator import TokenGenerator
from engine.agent_factory import create_agent
from engine.metrics import calculate_equilibrium_profit


def run_llm_episode(
    llm_agent_type: str = "GPT4-mini",
    opponent_pool: list[str] = None,
    num_buyers: int = 4,
    num_sellers: int = 4,
    num_tokens: int = 4,
    num_times: int = 50,  # Shorter episodes to save API costs
    price_min: int = 1,
    price_max: int = 100,
    seed: int = 42,
    llm_is_buyer: bool = True
) -> dict:
    """
    Run a single episode with LLM agent vs mixed opponents.
    """
    if opponent_pool is None:
        opponent_pool = ["ZIC", "Kaplan", "GD", "Lin"]

    rng = np.random.default_rng(seed)
    token_gen = TokenGenerator(1111, num_tokens, seed)

    buyers = []
    sellers = []
    llm_agent = None

    # Create buyers
    for i in range(num_buyers):
        pid = i + 1
        token_gen.new_round()
        tokens = token_gen.generate_tokens(True)

        if llm_is_buyer and i == 0:
            # First buyer is our LLM agent
            agent = create_agent(
                llm_agent_type, pid, True, num_tokens, tokens,
                price_min=price_min, price_max=price_max,
                num_times=num_times, seed=rng.integers(0, 100000)
            )
            llm_agent = agent
        else:
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

        if not llm_is_buyer and i == 0:
            agent = create_agent(
                llm_agent_type, pid, False, num_tokens, tokens,
                price_min=price_min, price_max=price_max,
                num_times=num_times, seed=rng.integers(0, 100000)
            )
            llm_agent = agent
        else:
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

    llm_profit = llm_agent.period_profit if llm_agent else 0
    llm_trades = llm_agent.num_trades if llm_agent else 0

    return {
        "llm_profit": llm_profit,
        "llm_trades": llm_trades,
        "market_efficiency": efficiency,
        "total_profit": actual_profit,
        "max_profit": max_profit
    }


def evaluate_llm(
    agent_type: str = "GPT4-mini",
    num_episodes: int = 10,  # Fewer episodes to save costs
    seed: int = 42
) -> dict:
    """
    Evaluate an LLM agent type over multiple episodes.
    """
    profits = []
    trades = []
    efficiencies = []

    for ep in range(num_episodes):
        print(f"  Episode {ep+1}/{num_episodes}...", end=" ", flush=True)
        try:
            result = run_llm_episode(
                llm_agent_type=agent_type,
                seed=seed + ep,
                llm_is_buyer=True
            )
            profits.append(result["llm_profit"])
            trades.append(result["llm_trades"])
            efficiencies.append(result["market_efficiency"])
            print(f"profit={result['llm_profit']:.2f}")
        except Exception as e:
            print(f"ERROR: {e}")

    if not profits:
        return {"error": "No successful episodes"}

    return {
        "avg_profit": float(np.mean(profits)),
        "std_profit": float(np.std(profits)),
        "avg_trades": float(np.mean(trades)),
        "avg_efficiency": float(np.mean(efficiencies)),
        "num_episodes": len(profits)
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate LLM agents vs Mixed")
    parser.add_argument("--model", type=str, default="GPT4-mini",
                       choices=["GPT4-mini", "GPT4", "GPT3.5"],
                       help="LLM model to test")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes (default: 10 to save costs)")
    parser.add_argument("--output", type=str, default="./results/llm_vs_mixed.json",
                       help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("=" * 60)
    print("LLM VS MIXED EVALUATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    print(f"\nEvaluating {args.model}...")
    results = evaluate_llm(
        agent_type=args.model,
        num_episodes=args.episodes,
        seed=args.seed
    )

    if "error" not in results:
        # Calculate ratio vs ZIC baseline (2.26 from legacy results)
        zic_baseline = 2.26
        results["profit_ratio_vs_zic"] = results["avg_profit"] / zic_baseline

        print(f"\n{'-' * 40}")
        print(f"Results for {args.model}:")
        print(f"  Avg profit: {results['avg_profit']:.2f} +/- {results['std_profit']:.2f}")
        print(f"  Avg trades: {results['avg_trades']:.1f}")
        print(f"  Avg efficiency: {results['avg_efficiency']:.2%}")
        print(f"  Ratio vs ZIC: {results['profit_ratio_vs_zic']:.2f}x")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "episodes": args.episodes,
        "results": results
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
