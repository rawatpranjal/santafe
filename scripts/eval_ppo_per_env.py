#!/usr/bin/env python3
"""
Evaluate PPO models that were trained per-environment.

Each PPO_ENV model is evaluated ONLY on its matching environment.
This is the methodologically correct approach since PPO cannot generalize
across market configurations.

Usage:
    python scripts/eval_ppo_per_env.py
    python scripts/eval_ppo_per_env.py --env BASE BBBS
    python scripts/eval_ppo_per_env.py --seeds 10 --rounds 50
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.metrics import calculate_equilibrium_profit
from engine.token_generator import TokenGenerator

# Environment configurations - must match training configs
ENV_CONFIGS = {
    "BASE": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_base/best_model/best_model.zip",
    },
    "BBBS": {
        "num_buyers": 6,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_bbbs/best_model/best_model.zip",
    },
    "BSSS": {
        "num_buyers": 2,
        "num_sellers": 6,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_bsss/best_model/best_model.zip",
    },
    "EQL": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 5555,
        "model_path": "checkpoints/ppo_eql/best_model/best_model.zip",
    },
    "RAN": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 9999,
        "model_path": "checkpoints/ppo_ran/best_model/best_model.zip",
    },
    "SHRT": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 20,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_shrt/best_model/best_model.zip",
    },
    "TOK": {
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 1,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_tok/best_model/best_model.zip",
    },
    "SML": {
        "num_buyers": 2,
        "num_sellers": 2,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_sml/best_model/best_model.zip",
    },
    "PER": {  # Single period - uses BASE model (same RL training)
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 1,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_base/best_model/best_model.zip",
    },
    "LAD": {  # Low adaptivity - uses BASE model (same RL training)
        "num_buyers": 4,
        "num_sellers": 4,
        "num_tokens": 4,
        "num_periods": 10,
        "num_steps": 100,
        "gametype": 6453,
        "model_path": "checkpoints/ppo_base/best_model/best_model.zip",
    },
}


def run_market_round(env_config, buyers, sellers, seed):
    """Run a single market round and return metrics."""
    num_steps = env_config["num_steps"]
    num_periods = env_config["num_periods"]

    all_agents = buyers + sellers
    round_profits = {agent: 0 for agent in all_agents}
    trade_prices = []
    total_trades = 0

    for period in range(1, num_periods + 1):
        for agent in all_agents:
            agent.start_period(period)

        market = Market(
            num_buyers=len(buyers),
            num_sellers=len(sellers),
            num_times=num_steps,
            price_min=0,
            price_max=1000,
            buyers=buyers,
            sellers=sellers,
            seed=seed * 1000 + period,
        )

        for _ in range(num_steps):
            market.run_time_step()

        for agent in all_agents:
            round_profits[agent] += agent.period_profit

        period_prices = [p for p in market.orderbook.trade_price if p > 0]
        trade_prices.extend(period_prices)
        total_trades += len(period_prices)

    # Calculate efficiency
    all_buyer_values = [v for b in buyers for v in b.valuations]
    all_seller_costs = [c for s in sellers for c in s.valuations]
    max_surplus = calculate_equilibrium_profit(all_buyer_values, all_seller_costs)
    actual_surplus = sum(round_profits.values())
    efficiency = (actual_surplus / max_surplus * 100) if max_surplus > 0 else 0

    # Price volatility
    if len(trade_prices) > 1:
        volatility = np.std(trade_prices) / np.mean(trade_prices) * 100
    else:
        volatility = 0

    return {
        "efficiency": efficiency,
        "volatility": volatility,
        "total_trades": total_trades,
        "profits": {type(a).__name__: round_profits[a] for a in all_agents},
    }


def eval_control(env_name, env_config, num_seeds, num_rounds):
    """Evaluate PPO vs 7 ZIC in control setting."""
    model_path = env_config["model_path"]

    if not Path(model_path).exists():
        print(f"  [SKIP] Model not found: {model_path}")
        return None

    results = {"efficiency": [], "volatility": [], "ppo_profit": [], "zic_profit": []}

    for seed in range(num_seeds):
        seed_eff, seed_vol, seed_ppo, seed_zic = [], [], [], []

        for round_idx in range(num_rounds):
            np.random.seed(seed * 10000 + round_idx)

            token_gen = TokenGenerator(
                env_config["gametype"], env_config["num_tokens"], seed * 10000 + round_idx
            )
            token_gen.new_round()

            n_buyers = env_config["num_buyers"]
            n_sellers = env_config["num_sellers"]

            # Create buyers: 1 PPO + (n-1) ZIC
            buyers = []

            # PPO buyer
            ppo_tokens = token_gen.generate_tokens(True)
            ppo_buyer = create_agent(
                "PPO",
                1,
                True,
                env_config["num_tokens"],
                ppo_tokens,
                model_path=model_path,
                seed=seed * 10000 + round_idx,
                num_times=env_config["num_steps"],
                num_buyers=n_buyers,
                num_sellers=n_sellers,
                price_min=0,
                price_max=1000,
            )
            buyers.append(ppo_buyer)

            # ZIC buyers
            for i in range(1, n_buyers):
                tokens = token_gen.generate_tokens(True)
                agent = create_agent(
                    "ZIC",
                    i + 1,
                    True,
                    env_config["num_tokens"],
                    tokens,
                    seed=seed * 10000 + round_idx + i,
                    num_times=env_config["num_steps"],
                    num_buyers=n_buyers,
                    num_sellers=n_sellers,
                    price_min=0,
                    price_max=1000,
                )
                buyers.append(agent)

            # Create sellers: all ZIC
            sellers = []
            for i in range(n_sellers):
                tokens = token_gen.generate_tokens(False)
                agent = create_agent(
                    "ZIC",
                    n_buyers + i + 1,
                    False,
                    env_config["num_tokens"],
                    tokens,
                    seed=seed * 10000 + round_idx + n_buyers + i,
                    num_times=env_config["num_steps"],
                    num_buyers=n_buyers,
                    num_sellers=n_sellers,
                    price_min=0,
                    price_max=1000,
                )
                sellers.append(agent)

            result = run_market_round(env_config, buyers, sellers, seed * 10000 + round_idx)

            seed_eff.append(result["efficiency"])
            seed_vol.append(result["volatility"])
            seed_ppo.append(result["profits"].get("PPOAgent", 0))
            seed_zic.append(sum(v for k, v in result["profits"].items() if "ZIC" in k))

        results["efficiency"].append(np.mean(seed_eff))
        results["volatility"].append(np.mean(seed_vol))
        results["ppo_profit"].append(np.mean(seed_ppo))
        results["zic_profit"].append(np.mean(seed_zic))

    return {
        "efficiency_mean": np.mean(results["efficiency"]),
        "efficiency_std": np.std(results["efficiency"]),
        "volatility_mean": np.mean(results["volatility"]),
        "volatility_std": np.std(results["volatility"]),
        "ppo_profit_mean": np.mean(results["ppo_profit"]),
        "zic_profit_mean": np.mean(results["zic_profit"]),
        "invasibility": np.mean(results["ppo_profit"])
        / max(1, abs(np.mean(results["zic_profit"]))),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate per-environment PPO models")
    parser.add_argument("--env", nargs="+", default=list(ENV_CONFIGS.keys()))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/ppo_per_env_{timestamp}"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PPO Per-Environment Evaluation (Control: PPO vs 7 ZIC)")
    print("=" * 70)
    print(f"Environments: {args.env}")
    print(f"Seeds: {args.seeds}, Rounds: {args.rounds}")
    print("=" * 70)

    all_results = {}

    for env_name in args.env:
        if env_name not in ENV_CONFIGS:
            print(f"\n[SKIP] Unknown environment: {env_name}")
            continue

        env_config = ENV_CONFIGS[env_name]
        print(f"\n--- {env_name} ---")
        print(
            f"  Config: {env_config['num_buyers']}B/{env_config['num_sellers']}S, "
            f"{env_config['num_tokens']} tokens, {env_config['num_steps']} steps, "
            f"gametype={env_config['gametype']}"
        )
        print(f"  Model: {env_config['model_path']}")

        result = eval_control(env_name, env_config, args.seeds, args.rounds)

        if result:
            all_results[env_name] = result
            print(
                f"  Efficiency: {result['efficiency_mean']:.1f}% +/- {result['efficiency_std']:.1f}%"
            )
            print(f"  Volatility: {result['volatility_mean']:.1f}%")
            print(f"  Invasibility: {result['invasibility']:.2f}x")

    # Save results
    with open(output_dir / "control_per_env.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: PPO Per-Environment Control Results")
    print("=" * 70)
    print(f"{'Env':<8} {'Efficiency':<15} {'Volatility':<12} {'Invasibility':<12}")
    print("-" * 50)

    for env_name, result in all_results.items():
        eff = f"{result['efficiency_mean']:.1f}+/-{result['efficiency_std']:.1f}%"
        vol = f"{result['volatility_mean']:.1f}%"
        inv = f"{result['invasibility']:.2f}x"
        print(f"{env_name:<8} {eff:<15} {vol:<12} {inv:<12}")

    print(f"\nResults saved to: {output_dir}")
    return all_results


if __name__ == "__main__":
    main()
