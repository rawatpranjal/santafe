"""
PPO vs Zero-Intelligence Baseline Experiment.

Tests PPO against the zero-intelligence hierarchy (ZI, ZIC, ZIP) to determine
where PPO ranks: can it beat ZIP?

Configuration:
- 4 buyers: ZI, ZIC, ZIP, PPO
- 4 sellers: ZI, ZIC, ZIP, ZIC (PPO is buyer-only due to role specialization)
- 50 rounds x 10 periods x 10 seeds
- BASE environment (gametype 6453)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator
from traders.rl.ppo_agent import PPOAgent

# Configuration
SEEDS = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
NUM_ROUNDS = 50
NUM_PERIODS = 10
PRICE_MIN = 1
PRICE_MAX = 1000

# BASE environment
ENV = {
    "gametype": 6453,
    "num_tokens": 4,
    "num_steps": 100,
    "num_buyers": 4,
    "num_sellers": 4,
}

# PPO model path
PPO_MODEL_PATH = "checkpoints/ppo_v8_mixed_competition/final_model.zip"

# Agent distribution
# Buyers: ZI, ZIC, ZIP, PPO (PPO is buyer-only)
# Sellers: ZI, ZIC, ZIP, ZIC (fourth seller is ZIC as control)
BUYER_TYPES = ["ZI", "ZIC", "ZIP", "PPO"]
SELLER_TYPES = ["ZI", "ZIC", "ZIP", "ZIC"]

# Strategies to track (PPO counted separately as buyer-only)
STRATEGIES = ["ZI", "ZIC", "ZIP", "PPO"]


def run_single_seed(seed: int) -> dict:
    """Run a tournament with one seed."""
    num_tokens = ENV["num_tokens"]
    num_steps = ENV["num_steps"]
    gametype = ENV["gametype"]
    num_buyers = ENV["num_buyers"]
    num_sellers = ENV["num_sellers"]

    # Track profits by strategy
    type_profits = {s: 0.0 for s in STRATEGIES}
    type_counts = {s: 0 for s in STRATEGIES}

    token_gen = TokenGenerator(gametype, num_tokens, seed)

    for r in range(NUM_ROUNDS):
        token_gen.new_round()

        # Create buyers
        buyers = []
        for i, agent_type in enumerate(BUYER_TYPES):
            player_id = i + 1
            vals = token_gen.generate_tokens(is_buyer=True)

            if agent_type == "PPO":
                agent = create_agent(
                    agent_type,
                    player_id=player_id,
                    is_buyer=True,
                    num_tokens=num_tokens,
                    valuations=vals,
                    price_min=PRICE_MIN,
                    price_max=PRICE_MAX,
                    num_times=num_steps,
                    seed=seed + player_id,
                    model_path=PPO_MODEL_PATH,
                )
            else:
                agent = create_agent(
                    agent_type,
                    player_id=player_id,
                    is_buyer=True,
                    num_tokens=num_tokens,
                    valuations=vals,
                    price_min=PRICE_MIN,
                    price_max=PRICE_MAX,
                    num_times=num_steps,
                    seed=seed + player_id,
                )
            agent.start_round(vals)
            buyers.append(agent)

        # Create sellers (no PPO - it's buyer-only)
        sellers = []
        for i, agent_type in enumerate(SELLER_TYPES):
            player_id = num_buyers + i + 1
            costs = token_gen.generate_tokens(is_buyer=False)
            agent = create_agent(
                agent_type,
                player_id=player_id,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=costs,
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                num_times=num_steps,
                seed=seed + player_id,
            )
            agent.start_round(costs)
            sellers.append(agent)

        # Run periods
        for p in range(NUM_PERIODS):
            market = Market(
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                num_times=num_steps,
                price_min=PRICE_MIN,
                price_max=PRICE_MAX,
                buyers=buyers,
                sellers=sellers,
            )
            market.set_period(r + 1, p + 1)

            # Inject orderbook into PPO agent
            for buyer in buyers:
                if isinstance(buyer, PPOAgent):
                    buyer.set_orderbook(market.orderbook)

            for agent in buyers + sellers:
                agent.start_period(p + 1)

            while market.current_time < market.num_times:
                market.run_time_step()

            for agent in buyers + sellers:
                agent.end_period()

        # Collect profits
        for i, agent in enumerate(buyers):
            agent_type = BUYER_TYPES[i]
            type_profits[agent_type] += agent.total_profit
            if agent_type != "PPO":
                type_counts[agent_type] += 1
            else:
                type_counts["PPO"] += 1

        for i, agent in enumerate(sellers):
            agent_type = SELLER_TYPES[i]
            # Only count ZI, ZIC, ZIP from sellers (no PPO sellers)
            if agent_type in ["ZI", "ZIC", "ZIP"]:
                type_profits[agent_type] += agent.total_profit
                type_counts[agent_type] += 1

    # Normalize by agent count per type
    normalized_profits = {}
    for s in STRATEGIES:
        if type_counts[s] > 0:
            normalized_profits[s] = type_profits[s] / type_counts[s]
        else:
            normalized_profits[s] = 0.0

    return normalized_profits


def main():
    """Run the full experiment."""
    print("=" * 70)
    print("PPO vs ZERO-INTELLIGENCE BASELINE EXPERIMENT")
    print("=" * 70)
    print(f"Buyers: {BUYER_TYPES}")
    print(f"Sellers: {SELLER_TYPES}")
    print(f"Seeds: {len(SEEDS)}")
    print(f"Rounds per seed: {NUM_ROUNDS}, Periods per round: {NUM_PERIODS}")
    print(f"PPO Model: {PPO_MODEL_PATH}")
    print("=" * 70)

    overall_start = time.time()

    all_profits = {s: [] for s in STRATEGIES}
    all_ranks = {s: [] for s in STRATEGIES}

    for seed in SEEDS:
        print(f"\nSeed {seed}: ", end="", flush=True)
        seed_start = time.time()

        profits = run_single_seed(seed)
        seed_elapsed = time.time() - seed_start
        print(f"{seed_elapsed:.1f}s", end=" ", flush=True)

        for s in STRATEGIES:
            all_profits[s].append(profits[s])

        # Calculate ranks for this seed
        sorted_by_profit = sorted(profits.items(), key=lambda x: -x[1])
        for rank, (strat, _) in enumerate(sorted_by_profit, 1):
            all_ranks[strat].append(rank)

        # Print rankings for this seed
        print(f"[{', '.join([f'{s[0]}:{r}' for s, r in sorted_by_profit])}]")

    total_elapsed = time.time() - overall_start
    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (10 seeds, 50 rounds each)")
    print("=" * 70)

    print("\n### Profit Table")
    print(f"{'Strategy':<10} {'Mean Profit':>15} {'Std':>10} {'Mean Rank':>12}")
    print("-" * 50)

    # Sort by mean profit
    results = []
    for s in STRATEGIES:
        mean_p = np.mean(all_profits[s])
        std_p = np.std(all_profits[s])
        mean_r = np.mean(all_ranks[s])
        results.append((s, mean_p, std_p, mean_r))

    results.sort(key=lambda x: -x[1])  # Sort by profit descending

    for s, mean_p, std_p, mean_r in results:
        print(f"{s:<10} {mean_p:>15,.0f} {std_p:>10,.0f} {mean_r:>12.2f}")

    print("\n### Rankings by seed")
    header = f"{'Seed':<8}"
    for s in STRATEGIES:
        header += f"{s:>8}"
    print(header)
    for i, seed in enumerate(SEEDS):
        row = f"{seed:<8}"
        for s in STRATEGIES:
            row += f"{all_ranks[s][i]:>8}"
        print(row)

    print("\n### Average Rank")
    for s in STRATEGIES:
        print(f"  {s}: {np.mean(all_ranks[s]):.2f}")

    # Save results to JSON
    results_dir = Path("results/ppo_vs_zi")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_data = {
        "config": {
            "seeds": SEEDS,
            "num_rounds": NUM_ROUNDS,
            "num_periods": NUM_PERIODS,
            "buyer_types": BUYER_TYPES,
            "seller_types": SELLER_TYPES,
            "ppo_model": PPO_MODEL_PATH,
            "environment": ENV,
        },
        "results": {
            s: {
                "mean_profit": float(np.mean(all_profits[s])),
                "std_profit": float(np.std(all_profits[s])),
                "mean_rank": float(np.mean(all_ranks[s])),
                "profits_by_seed": [float(p) for p in all_profits[s]],
                "ranks_by_seed": [int(r) for r in all_ranks[s]],
            }
            for s in STRATEGIES
        },
        "summary": {
            "winner": results[0][0],
            "winner_profit": float(results[0][1]),
            "ppo_rank": float(np.mean(all_ranks["PPO"])),
            "zip_rank": float(np.mean(all_ranks["ZIP"])),
            "ppo_beats_zip": float(np.mean(all_ranks["PPO"])) < float(np.mean(all_ranks["ZIP"])),
        },
    }

    with open(results_dir / "results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_dir / 'results.json'}")

    # Key finding
    print("\n" + "=" * 70)
    print("KEY FINDING")
    print("=" * 70)
    ppo_rank = np.mean(all_ranks["PPO"])
    zip_rank = np.mean(all_ranks["ZIP"])
    if ppo_rank < zip_rank:
        print(f"PPO (rank {ppo_rank:.2f}) BEATS ZIP (rank {zip_rank:.2f})")
        print("RL exceeds hand-crafted adaptive learning!")
    elif ppo_rank > zip_rank:
        print(f"ZIP (rank {zip_rank:.2f}) BEATS PPO (rank {ppo_rank:.2f})")
        print("Hand-crafted adaptive learning remains superior.")
    else:
        print(f"PPO and ZIP are TIED (rank {ppo_rank:.2f})")


if __name__ == "__main__":
    main()
