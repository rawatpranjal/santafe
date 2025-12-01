#!/usr/bin/env python3
"""
Stress test for LLM dashboard prompts with GPT-4 Turbo.

Runs 1 round × N periods × 100 steps to validate the dashboard prompt works.
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


def run_stress_test(
    model: str = "gpt-4-turbo",
    prompt_style: str = "dashboard",
    num_steps: int = 100,
    num_periods: int = 1,
    num_tokens: int = 4,
    gametype: int = 6453,
    seed: int = 42,
    reasoning_effort: str | None = None,
    output_dir: str | None = None,
    opponent: str = "ZIC",
) -> dict:
    """
    Run stress test: 1 LLM buyer + 3 opponent buyers, 4 opponent sellers.
    1 round × num_periods periods × num_steps.
    Opponent can be ZIC or ZIP.
    """
    print(f"\n{'='*60}")
    print("STRESS TEST: GPT-4 Turbo Dashboard Prompt")
    print(f"{'='*60}")
    print(f"Model: {model}")
    print(f"Prompt Style: {prompt_style}")
    print(f"Config: 1 round × {num_periods} periods × {num_steps} steps")
    print(f"{'='*60}\n")

    num_buyers = 4
    num_sellers = 4
    price_min = 1
    price_max = 1000

    rng = np.random.default_rng(seed)

    # Create token generator
    token_gen = TokenGenerator(gametype, num_tokens, seed)
    token_gen.new_round()

    # Create agents
    agents = []
    buyer_valuations = {}
    seller_costs = {}

    # Map model parameter to agent type
    model_to_agent = {
        "gpt-4-turbo": "GPT4-turbo",
        "gpt-4o": "GPT4",
        "gpt-4o-mini": "GPT4-mini",
        "gpt-5-nano": "GPT5-nano",
        "gpt-4.1": "GPT4.1",
        "gpt-4.1-mini": "GPT4.1-mini",
        "gpt-4.1-nano": "GPT4.1-nano",
        "gpt-3.5-turbo": "GPT3.5",
        "o4-mini": "O4-mini",
    }
    llm_agent_type = model_to_agent.get(model, model)  # fallback to model name if not in map
    buyer_types = [llm_agent_type, opponent, opponent, opponent]
    seller_types = [opponent, opponent, opponent, opponent]

    # Buyers
    for i, agent_type in enumerate(buyer_types):
        pid = i + 1
        tokens = token_gen.generate_tokens(True)
        buyer_valuations[pid] = tokens

        kwargs = {"num_buyers": num_buyers, "num_sellers": num_sellers}
        if agent_type.startswith("GPT") or agent_type.startswith("O4"):
            kwargs["prompt_style"] = prompt_style
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort
            if output_dir:
                kwargs["output_dir"] = output_dir

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
        print(f"Buyer {pid}: {agent_type}, tokens={tokens}")

    # Sellers
    for i, agent_type in enumerate(seller_types):
        pid = num_buyers + i + 1
        tokens = token_gen.generate_tokens(False)
        seller_costs[i + 1] = tokens

        kwargs = {"num_buyers": num_buyers, "num_sellers": num_sellers}

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
        print(f"Seller {pid}: {agent_type}, tokens={tokens}")

    # Calculate max surplus for efficiency
    buyer_vals_list = [buyer_valuations[a.player_id] for a in agents if a.is_buyer]
    seller_costs_list = [seller_costs[a.player_id - num_buyers] for a in agents if not a.is_buyer]
    max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)
    print(f"\nMax surplus per period: {max_surplus}")

    # Track per-period results
    period_results = []
    llm_wins = 0
    llm_total_profit = 0
    zic_total_profit = 0

    # Run all periods
    for period in range(1, num_periods + 1):
        print(f"\n--- Running Period {period}/{num_periods} ---")

        # Create market for this period
        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            price_min=price_min,
            price_max=price_max,
            num_times=num_steps,
            buyers=[a for a in agents if a.is_buyer],
            sellers=[a for a in agents if not a.is_buyer],
            seed=seed + period,
        )

        # Start period
        for a in agents:
            a.start_period(period)

        # Run market with progress
        print("Running steps: ", end="", flush=True)
        for t in range(num_steps):
            if t % 10 == 0:
                print(f"{t}", end=".", flush=True)
            market.run_time_step()
        print(" done!")

        # End period
        for a in agents:
            a.end_period()

        # Extract trades
        trades = extract_trades_from_orderbook(market.orderbook, num_steps)

        # Calculate efficiency
        actual_surplus = calculate_actual_surplus(trades, buyer_valuations, seller_costs)
        efficiency = (actual_surplus / max_surplus * 100) if max_surplus > 0 else 0

        # Calculate volatility
        prices = get_transaction_prices(market.orderbook, num_steps)
        if len(prices) > 1:
            volatility = float(np.std(prices) / np.mean(prices) * 100) if np.mean(prices) > 0 else 0
        else:
            volatility = 0

        # Get LLM agent stats
        llm_agent = agents[0]
        llm_profit = llm_agent.period_profit
        llm_trades = llm_agent.num_trades

        # Get ZIC average profit (buyers only for comparison)
        zic_buyer_profits = [a.period_profit for a in agents[1:4]]
        zic_avg_profit = np.mean(zic_buyer_profits) if zic_buyer_profits else 0

        # Calculate ratio
        ratio = llm_profit / zic_avg_profit if zic_avg_profit > 0 else float("inf")

        # Track wins
        if llm_profit > zic_avg_profit:
            llm_wins += 1
            win_str = "✓ WIN"
        else:
            win_str = "✗ LOSS"

        llm_total_profit += llm_profit
        zic_total_profit += zic_avg_profit

        # Print period summary
        print(
            f"  Trades: {len(trades)} | Eff: {efficiency:.1f}% | LLM: ${llm_profit} | ZIC: ${zic_avg_profit:.0f} | Ratio: {ratio:.2f}x {win_str}"
        )

        period_results.append(
            {
                "period": period,
                "efficiency": efficiency,
                "volatility": volatility,
                "num_trades": len(trades),
                "llm_profit": llm_profit,
                "llm_trades": llm_trades,
                "zic_avg_profit": zic_avg_profit,
                "profit_ratio": ratio,
                "llm_won": bool(llm_profit > zic_avg_profit),
            }
        )

    # Get final invalid rate
    llm_agent = agents[0]
    invalid_rate = (
        llm_agent.get_invalid_action_rate() if hasattr(llm_agent, "get_invalid_action_rate") else 0
    )

    # Aggregate results
    avg_efficiency = np.mean([r["efficiency"] for r in period_results])
    avg_volatility = np.mean([r["volatility"] for r in period_results])
    total_trades = sum([r["num_trades"] for r in period_results])
    avg_ratio = llm_total_profit / zic_total_profit if zic_total_profit > 0 else float("inf")

    # Print final results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Periods: {num_periods}")
    print(f"Average Efficiency: {avg_efficiency:.1f}%")
    print(f"Average Volatility: {avg_volatility:.1f}%")
    print(f"Total Trades: {total_trades}")
    print("\nLLM Agent Performance:")
    print(f"  Total Profit: ${llm_total_profit}")
    print(f"  ZIC Total Profit: ${zic_total_profit:.0f}")
    print(f"  Overall Ratio: {avg_ratio:.2f}x")
    print(f"  Win Rate: {llm_wins}/{num_periods} periods ({100*llm_wins/num_periods:.0f}%)")
    print(f"  Invalid Action Rate: {invalid_rate:.1f}%")

    # Show per-period breakdown
    print(f"\n{'='*60}")
    print("PER-PERIOD BREAKDOWN")
    print(f"{'='*60}")
    print(f"{'Period':<8} {'LLM':<10} {'ZIC':<10} {'Ratio':<10} {'Result':<8}")
    print("-" * 46)
    for r in period_results:
        result_str = "WIN" if r["llm_won"] else "LOSS"
        print(
            f"{r['period']:<8} ${r['llm_profit']:<9} ${r['zic_avg_profit']:<9.0f} {r['profit_ratio']:<9.2f}x {result_str}"
        )

    results = {
        "avg_efficiency": avg_efficiency,
        "avg_volatility": avg_volatility,
        "total_trades": total_trades,
        "llm_total_profit": llm_total_profit,
        "zic_total_profit": zic_total_profit,
        "overall_ratio": avg_ratio,
        "win_rate": llm_wins / num_periods,
        "llm_wins": llm_wins,
        "invalid_rate": invalid_rate,
        "period_results": period_results,
    }

    # Save results
    output_path = Path(f"results/stress_test_llm_dashboard_{num_periods}periods.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt_style": prompt_style,
                "config": {
                    "num_rounds": 1,
                    "num_periods": num_periods,
                    "num_steps": num_steps,
                    "num_tokens": num_tokens,
                },
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", type=int, default=10, help="Number of periods")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-turbo",
        help="LLM model to use (e.g., gpt-4-turbo, gpt-4.1-nano, o4-mini)",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Reasoning effort for o-series models",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save prompts/responses for debugging",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="dashboard",
        choices=["minimal", "deep", "dashboard", "constraints", "dense"],
        help="Prompt style",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--opponent",
        type=str,
        default="ZIC",
        choices=["ZIC", "ZIP"],
        help="Opponent trader type (default: ZIC)",
    )
    args = parser.parse_args()

    run_stress_test(
        model=args.model,
        num_periods=args.periods,
        reasoning_effort=args.reasoning,
        output_dir=args.output_dir,
        prompt_style=args.prompt_style,
        seed=args.seed,
        opponent=args.opponent,
    )
