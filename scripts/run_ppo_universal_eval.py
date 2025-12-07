#!/usr/bin/env python3
"""
Universal PPO Evaluation Script.

Evaluates the trained universal PPO model against:
1. Pure opponents (ZI, ZIC, ZIC2, ZIP, ZIP2) - 5 tests per env
2. Mixed opponents (random from pool) - 1 test per env

Across all 10 Santa Fe environments.

Usage:
    python scripts/run_ppo_universal_eval.py
    python scripts/run_ppo_universal_eval.py --model checkpoints/ppo_universal/final_model.zip
    python scripts/run_ppo_universal_eval.py --seeds 10 --rounds 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))


from engine.agent_factory import create_agent
from engine.market import Market
from engine.metrics import calculate_equilibrium_profit
from engine.token_generator import TokenGenerator
from envs.enhanced_double_auction_env import OPPONENT_POOL, SANTA_FE_ENVIRONMENTS
from traders.rl.ppo_agent import PPOAgent


def run_tournament(
    env_name: str,
    opponent_type: str,  # "ZI", "ZIC", "ZIC2", "ZIP", "ZIP2", or "Mixed"
    model_path: str,
    seeds: list[int],
    num_rounds: int = 50,
    num_periods: int = 10,
) -> dict:
    """
    Run PPO vs opponents tournament.

    PPO is always buyer 1. Other 7 agents are opponents.
    """
    env_config = SANTA_FE_ENVIRONMENTS[env_name]
    gametype = env_config["gametype"]
    max_price = env_config["max_price"]
    num_buyers = env_config["num_buyers"]
    num_sellers = env_config["num_sellers"]
    num_tokens = env_config["num_tokens"]
    num_steps = env_config["max_steps"]

    all_profits = []
    all_ranks = []
    all_efficiencies = []
    all_trades = []

    for seed in seeds:
        np.random.seed(seed)
        token_gen = TokenGenerator(gametype, num_tokens, seed)

        seed_profits = []
        seed_ranks = []
        seed_efficiencies = []
        seed_trades = []

        for round_idx in range(num_rounds):
            token_gen.new_round()

            # Create agents
            agents = []
            ppo_agent = None

            # Determine opponent types for this round
            if opponent_type == "Mixed":
                opp_types = [
                    np.random.choice(OPPONENT_POOL) for _ in range(num_buyers + num_sellers - 1)
                ]
            else:
                opp_types = [opponent_type] * (num_buyers + num_sellers - 1)

            opp_idx = 0

            # Create buyers
            for i in range(num_buyers):
                pid = i + 1
                tokens = token_gen.generate_tokens(True)

                if pid == 1:  # PPO is buyer 1
                    ppo_agent = PPOAgent(
                        player_id=pid,
                        is_buyer=True,
                        num_tokens=num_tokens,
                        valuations=tokens,
                        model_path=model_path,
                        max_price=max_price,
                        min_price=0,
                        max_steps=num_steps,
                    )
                    # Set env context on obs generator
                    ppo_agent.obs_gen.set_env_context(
                        num_buyers=num_buyers,
                        num_sellers=num_sellers,
                        num_tokens=num_tokens,
                        max_steps=num_steps,
                        gametype=gametype,
                    )
                    agents.append(ppo_agent)
                else:
                    agent = create_agent(
                        opp_types[opp_idx],
                        pid,
                        True,
                        num_tokens,
                        tokens,
                        seed=seed + pid + round_idx,
                        num_times=num_steps,
                        price_min=0,
                        price_max=max_price,
                    )
                    agents.append(agent)
                    opp_idx += 1

            # Create sellers
            for i in range(num_sellers):
                pid = num_buyers + i + 1
                tokens = token_gen.generate_tokens(False)

                agent = create_agent(
                    opp_types[opp_idx],
                    pid,
                    False,
                    num_tokens,
                    tokens,
                    seed=seed + pid + round_idx,
                    num_times=num_steps,
                    price_min=0,
                    price_max=max_price,
                )
                agents.append(agent)
                opp_idx += 1

            # Start round
            for agent in agents:
                agent.start_round(agent.valuations)

            # Run periods
            for period_idx in range(num_periods):
                buyers = [a for a in agents if a.is_buyer]
                sellers = [a for a in agents if not a.is_buyer]

                market = Market(
                    num_buyers=num_buyers,
                    num_sellers=num_sellers,
                    num_times=num_steps,
                    price_min=0,
                    price_max=max_price,
                    buyers=buyers,
                    sellers=sellers,
                    seed=seed + round_idx + period_idx,
                )
                market.set_period(round_idx + 1, period_idx + 1)

                # Inject orderbook into PPO agent
                ppo_agent.set_orderbook(market.orderbook)

                # Start period
                for agent in agents:
                    agent.start_period(period_idx + 1)

                # Run period
                while market.current_time < num_steps:
                    market.run_time_step()

                # End period
                for agent in agents:
                    agent.end_period()

                # Calculate efficiency
                all_vals = [v for b in buyers for v in b.valuations]
                all_costs = [c for s in sellers for c in s.valuations]
                max_surplus = calculate_equilibrium_profit(all_vals, all_costs)
                actual_surplus = sum(a.period_profit for a in agents)
                eff = actual_surplus / max_surplus if max_surplus > 0 else 0
                seed_efficiencies.append(eff)

                # Count trades
                trades = sum(a.num_trades for a in buyers)
                seed_trades.append(trades)

            # End of round - collect profits
            ppo_profit = ppo_agent.total_profit
            seed_profits.append(ppo_profit)

            # Calculate rank among buyers
            buyer_profits = [(a.player_id, a.total_profit) for a in agents if a.is_buyer]
            buyer_profits.sort(key=lambda x: -x[1])
            ppo_rank = next(i + 1 for i, (pid, _) in enumerate(buyer_profits) if pid == 1)
            seed_ranks.append(ppo_rank)

        all_profits.extend(seed_profits)
        all_ranks.extend(seed_ranks)
        all_efficiencies.extend(seed_efficiencies)
        all_trades.extend(seed_trades)

    return {
        "env": env_name,
        "opponent": opponent_type,
        "ppo_profit_mean": float(np.mean(all_profits)),
        "ppo_profit_std": float(np.std(all_profits)),
        "ppo_rank_mean": float(np.mean(all_ranks)),
        "ppo_rank_std": float(np.std(all_ranks)),
        "market_efficiency_mean": float(np.mean(all_efficiencies)),
        "trades_per_period_mean": float(np.mean(all_trades)),
        "num_seeds": len(seeds),
        "num_rounds": num_rounds,
    }


def main():
    parser = argparse.ArgumentParser(description="Universal PPO Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/ppo_universal/final_model.zip",
        help="Path to trained model",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--rounds", type=int, default=50, help="Rounds per seed")
    parser.add_argument("--periods", type=int, default=10, help="Periods per round")
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=None,
        help="Specific envs to test (default: all)",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=None,
        help="Specific opponents to test (default: all + Mixed)",
    )
    args = parser.parse_args()

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first with: python scripts/train_ppo_universal.py")
        sys.exit(1)

    # Environments to test
    envs = args.envs or list(SANTA_FE_ENVIRONMENTS.keys())

    # Opponents to test
    opponents = args.opponents or ["ZI", "ZIC", "ZIC2", "ZIP", "ZIP2", "Mixed"]

    # Seeds
    seeds = list(range(42, 42 + args.seeds))

    print("=" * 70)
    print("UNIVERSAL PPO EVALUATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Environments: {envs}")
    print(f"Opponents: {opponents}")
    print(f"Seeds: {len(seeds)}, Rounds: {args.rounds}, Periods: {args.periods}")
    print("=" * 70)

    results = []

    for env_name in envs:
        print(f"\n--- {env_name} ---")
        for opp in opponents:
            print(f"  vs {opp}...", end=" ", flush=True)
            result = run_tournament(
                env_name=env_name,
                opponent_type=opp,
                model_path=str(model_path),
                seeds=seeds,
                num_rounds=args.rounds,
                num_periods=args.periods,
            )
            results.append(result)
            print(
                f"profit={result['ppo_profit_mean']:.0f}, "
                f"rank={result['ppo_rank_mean']:.1f}, "
                f"eff={result['market_efficiency_mean']:.2f}"
            )

    # Save results
    output_dir = Path("results/ppo_universal_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"results": results, "config": vars(args)}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY: PPO Rank by Environment and Opponent")
    print("=" * 70)
    print(f"{'Env':<8}", end="")
    for opp in opponents:
        print(f"{opp:>8}", end="")
    print()
    print("-" * 70)

    for env_name in envs:
        print(f"{env_name:<8}", end="")
        for opp in opponents:
            result = next(
                (r for r in results if r["env"] == env_name and r["opponent"] == opp), None
            )
            if result:
                print(f"{result['ppo_rank_mean']:>8.1f}", end="")
            else:
                print(f"{'N/A':>8}", end="")
        print()

    # Average across environments
    print("-" * 70)
    print(f"{'AVG':<8}", end="")
    for opp in opponents:
        opp_results = [r for r in results if r["opponent"] == opp]
        avg_rank = np.mean([r["ppo_rank_mean"] for r in opp_results])
        print(f"{avg_rank:>8.2f}", end="")
    print()

    print("\n" + "=" * 70)
    print("SUMMARY: PPO Profit by Environment and Opponent")
    print("=" * 70)
    print(f"{'Env':<8}", end="")
    for opp in opponents:
        print(f"{opp:>10}", end="")
    print()
    print("-" * 70)

    for env_name in envs:
        print(f"{env_name:<8}", end="")
        for opp in opponents:
            result = next(
                (r for r in results if r["env"] == env_name and r["opponent"] == opp), None
            )
            if result:
                print(f"{result['ppo_profit_mean']:>10.0f}", end="")
            else:
                print(f"{'N/A':>10}", end="")
        print()

    print("\nDone!")


if __name__ == "__main__":
    main()
