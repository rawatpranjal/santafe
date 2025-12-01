"""
PPO vs Zero-Intelligence Full Metrics Analysis.

Captures comprehensive metrics for PPO entry into ZI/ZIC/ZIP market:
- Market-level: efficiency, volatility, v-inefficiency, trades per period
- Individual: profit by strategy, profit dispersion
- Behavior: PPO trade timing, price positioning

Uses trained model from checkpoints/ppo_vs_zi_mix/final_model.zip
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.agent_factory import create_agent
from engine.efficiency import (
    calculate_allocative_efficiency,
    calculate_equilibrium_price,
    calculate_max_surplus,
    calculate_price_std_dev,
    calculate_smiths_alpha,
    calculate_v_inefficiency,
)
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

# PPO model (trained against ZIC/ZIP mix)
PPO_MODEL_PATH = "checkpoints/ppo_vs_zi_mix/final_model.zip"

# Market configurations to compare
MARKET_CONFIGS = {
    "ppo_mix": {
        "buyer_types": ["ZI", "ZIC", "ZIP", "PPO"],
        "seller_types": ["ZI", "ZIC", "ZIP", "ZIC"],
        "description": "PPO + ZI + ZIC + ZIP mixed market",
    },
    "zi_only": {
        "buyer_types": ["ZI", "ZI", "ZI", "ZI"],
        "seller_types": ["ZI", "ZI", "ZI", "ZI"],
        "description": "Pure ZI market (baseline)",
    },
    "zic_only": {
        "buyer_types": ["ZIC", "ZIC", "ZIC", "ZIC"],
        "seller_types": ["ZIC", "ZIC", "ZIC", "ZIC"],
        "description": "Pure ZIC market (baseline)",
    },
    "zip_only": {
        "buyer_types": ["ZIP", "ZIP", "ZIP", "ZIP"],
        "seller_types": ["ZIP", "ZIP", "ZIP", "ZIP"],
        "description": "Pure ZIP market (baseline)",
    },
}

STRATEGIES = ["ZI", "ZIC", "ZIP", "PPO"]


def run_single_period(buyers: list, sellers: list, market: Market, num_steps: int) -> dict:
    """Run a single period and extract detailed metrics."""
    # Run period
    while market.current_time < num_steps:
        market.run_time_step()

    ob = market.orderbook

    # Get valuations as lists for max_surplus
    buyer_vals_list = [list(b.valuations) for b in buyers]
    seller_costs_list = [list(s.valuations) for s in sellers]

    # Calculate max possible surplus
    max_surplus = calculate_max_surplus(buyer_vals_list, seller_costs_list)

    # Calculate actual surplus directly from agent period profits (more robust)
    # Actual surplus = sum of all agents' profits this period
    actual_surplus = 0.0
    for buyer in buyers:
        actual_surplus += buyer.period_profit
    for seller in sellers:
        actual_surplus += seller.period_profit

    efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

    # Count trades directly from orderbook
    num_trades = 0
    for t in range(1, num_steps + 1):
        if int(ob.trade_price[t]) > 0:
            num_trades += 1

    # V-inefficiency: missed surplus from untraded intra-marginal units
    # Also compute max_trades for reference
    all_buyer_vals = sorted([v for vals in buyer_vals_list for v in vals], reverse=True)
    all_seller_costs = sorted([c for costs in seller_costs_list for c in costs])
    max_trades = 0
    for i in range(min(len(all_buyer_vals), len(all_seller_costs))):
        if all_buyer_vals[i] > all_seller_costs[i]:
            max_trades += 1
        else:
            break

    v_ineff = calculate_v_inefficiency(num_trades, buyer_vals_list, seller_costs_list)

    # Price metrics - get prices directly from orderbook
    prices = []
    for t in range(1, num_steps + 1):
        p = int(ob.trade_price[t])
        if p > 0:
            prices.append(p)

    price_std = calculate_price_std_dev(prices) if prices else 0.0
    eq_price = calculate_equilibrium_price(buyer_vals_list, seller_costs_list)
    price_volatility = (price_std / eq_price * 100) if eq_price > 0 else 0.0
    smiths_alpha = calculate_smiths_alpha(prices, eq_price) if prices else float("inf")

    return {
        "efficiency": efficiency,
        "v_inefficiency": v_ineff,
        "num_trades": num_trades,
        "max_trades": max_trades,
        "price_volatility": price_volatility,
        "price_std": price_std,
        "smiths_alpha": smiths_alpha,
        "equilibrium_price": eq_price,
        "actual_surplus": actual_surplus,
        "max_surplus": max_surplus,
        "trade_prices": prices,
    }


def run_market_config(config_name: str, seed: int) -> dict:
    """Run a complete tournament for one market configuration."""
    config = MARKET_CONFIGS[config_name]
    buyer_types = config["buyer_types"]
    seller_types = config["seller_types"]

    num_tokens = ENV["num_tokens"]
    num_steps = ENV["num_steps"]
    gametype = ENV["gametype"]
    num_buyers = ENV["num_buyers"]
    num_sellers = ENV["num_sellers"]

    # Track profits by strategy
    type_profits = {s: 0.0 for s in STRATEGIES if s in buyer_types + seller_types}
    type_counts = {s: 0 for s in STRATEGIES if s in buyer_types + seller_types}

    # Track market metrics per period
    all_metrics = []

    # Track PPO-specific behavior (if PPO in market)
    ppo_trade_times = []  # When in period PPO trades
    ppo_trade_prices = []  # At what prices PPO trades

    token_gen = TokenGenerator(gametype, num_tokens, seed)

    for r in range(NUM_ROUNDS):
        token_gen.new_round()

        # Create buyers
        buyers = []
        for i, agent_type in enumerate(buyer_types):
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

        # Create sellers
        sellers = []
        for i, agent_type in enumerate(seller_types):
            player_id = num_buyers + i + 1
            costs = token_gen.generate_tokens(is_buyer=False)

            if agent_type == "PPO":
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
                    model_path=PPO_MODEL_PATH,
                )
            else:
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

            # Track PPO trade positions before running
            ppo_buyer_idx = None
            for i, bt in enumerate(buyer_types):
                if bt == "PPO":
                    ppo_buyer_idx = i
                    break

            ppo_trades_before = 0
            if ppo_buyer_idx is not None:
                ppo_trades_before = buyers[ppo_buyer_idx].num_trades

            # Run period and get metrics
            period_metrics = run_single_period(buyers, sellers, market, num_steps)
            all_metrics.append(period_metrics)

            # Track PPO trade behavior
            if ppo_buyer_idx is not None:
                ob = market.orderbook
                ppo_trades_after = buyers[ppo_buyer_idx].num_trades

                # Find when PPO traded by checking trade records
                for t in range(1, num_steps + 1):
                    price = int(ob.trade_price[t])
                    if price > 0:
                        # Check if PPO buyer traded at this time
                        ppo_id = ppo_buyer_idx + 1
                        if int(ob.num_buys[ppo_id, t]) > int(
                            ob.num_buys[ppo_id, t - 1] if t > 1 else 0
                        ):
                            ppo_trade_times.append(t)
                            ppo_trade_prices.append(price)

            for agent in buyers + sellers:
                agent.end_period()

        # Collect profits
        for i, agent in enumerate(buyers):
            agent_type = buyer_types[i]
            if agent_type in type_profits:
                type_profits[agent_type] += agent.total_profit
                type_counts[agent_type] += 1

        for i, agent in enumerate(sellers):
            agent_type = seller_types[i]
            if agent_type in type_profits:
                type_profits[agent_type] += agent.total_profit
                type_counts[agent_type] += 1

    # Aggregate metrics
    agg_metrics = {
        "efficiency_mean": float(np.mean([m["efficiency"] for m in all_metrics])),
        "efficiency_std": float(np.std([m["efficiency"] for m in all_metrics])),
        "v_inefficiency_mean": float(np.mean([m["v_inefficiency"] for m in all_metrics])),
        "v_inefficiency_std": float(np.std([m["v_inefficiency"] for m in all_metrics])),
        "trades_per_period_mean": float(np.mean([m["num_trades"] for m in all_metrics])),
        "trades_per_period_std": float(np.std([m["num_trades"] for m in all_metrics])),
        "price_volatility_mean": float(np.mean([m["price_volatility"] for m in all_metrics])),
        "price_volatility_std": float(np.std([m["price_volatility"] for m in all_metrics])),
        "smiths_alpha_mean": float(
            np.mean([m["smiths_alpha"] for m in all_metrics if m["smiths_alpha"] != float("inf")])
        ),
    }

    # Normalize profits
    normalized_profits = {}
    for s in type_profits:
        if type_counts[s] > 0:
            normalized_profits[s] = type_profits[s] / type_counts[s]
        else:
            normalized_profits[s] = 0.0

    # PPO behavior metrics
    ppo_behavior = {}
    if ppo_trade_times:
        ppo_behavior = {
            "avg_trade_time": float(np.mean(ppo_trade_times)),
            "trade_time_std": float(np.std(ppo_trade_times)),
            "avg_trade_price": float(np.mean(ppo_trade_prices)),
            "trade_price_std": float(np.std(ppo_trade_prices)),
            "num_trades": len(ppo_trade_times),
            "early_trades_pct": float(
                sum(1 for t in ppo_trade_times if t <= 30) / len(ppo_trade_times) * 100
            ),
            "late_trades_pct": float(
                sum(1 for t in ppo_trade_times if t >= 70) / len(ppo_trade_times) * 100
            ),
        }

    return {
        "market_metrics": agg_metrics,
        "profits": normalized_profits,
        "ppo_behavior": ppo_behavior,
    }


def main():
    """Run full metrics analysis."""
    print("=" * 70)
    print("PPO vs ZERO-INTELLIGENCE FULL METRICS ANALYSIS")
    print("=" * 70)
    print(f"Seeds: {len(SEEDS)}")
    print(f"Rounds per seed: {NUM_ROUNDS}, Periods per round: {NUM_PERIODS}")
    print(f"PPO Model: {PPO_MODEL_PATH}")
    print("=" * 70)

    results_dir = Path("results/ppo_vs_zi_metrics")
    results_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()
    all_results = {}

    for config_name, config in MARKET_CONFIGS.items():
        print(f"\n{'=' * 50}")
        print(f"Running: {config['description']}")
        print("=" * 50)

        config_results = {
            "efficiency": [],
            "v_inefficiency": [],
            "trades_per_period": [],
            "price_volatility": [],
            "smiths_alpha": [],
            "profits": {
                s: [] for s in STRATEGIES if s in config["buyer_types"] + config["seller_types"]
            },
            "ranks": {
                s: [] for s in STRATEGIES if s in config["buyer_types"] + config["seller_types"]
            },
            "ppo_behavior": [],
        }

        for seed in SEEDS:
            print(f"  Seed {seed}: ", end="", flush=True)
            seed_start = time.time()

            result = run_market_config(config_name, seed)

            # Collect metrics
            config_results["efficiency"].append(result["market_metrics"]["efficiency_mean"])
            config_results["v_inefficiency"].append(result["market_metrics"]["v_inefficiency_mean"])
            config_results["trades_per_period"].append(
                result["market_metrics"]["trades_per_period_mean"]
            )
            config_results["price_volatility"].append(
                result["market_metrics"]["price_volatility_mean"]
            )
            config_results["smiths_alpha"].append(result["market_metrics"]["smiths_alpha_mean"])

            for s, p in result["profits"].items():
                config_results["profits"][s].append(p)

            # Calculate ranks
            sorted_profits = sorted(result["profits"].items(), key=lambda x: -x[1])
            for rank, (strat, _) in enumerate(sorted_profits, 1):
                config_results["ranks"][strat].append(rank)

            if result["ppo_behavior"]:
                config_results["ppo_behavior"].append(result["ppo_behavior"])

            elapsed = time.time() - seed_start
            print(f"{elapsed:.1f}s")

        # Aggregate for this config
        all_results[config_name] = {
            "description": config["description"],
            "market_metrics": {
                "efficiency": {
                    "mean": float(np.mean(config_results["efficiency"])),
                    "std": float(np.std(config_results["efficiency"])),
                },
                "v_inefficiency": {
                    "mean": float(np.mean(config_results["v_inefficiency"])),
                    "std": float(np.std(config_results["v_inefficiency"])),
                },
                "trades_per_period": {
                    "mean": float(np.mean(config_results["trades_per_period"])),
                    "std": float(np.std(config_results["trades_per_period"])),
                },
                "price_volatility": {
                    "mean": float(np.mean(config_results["price_volatility"])),
                    "std": float(np.std(config_results["price_volatility"])),
                },
                "smiths_alpha": {
                    "mean": float(
                        np.mean([a for a in config_results["smiths_alpha"] if not np.isinf(a)])
                    ),
                    "std": float(
                        np.std([a for a in config_results["smiths_alpha"] if not np.isinf(a)])
                    ),
                },
            },
            "profits": {
                s: {
                    "mean": float(np.mean(config_results["profits"][s])),
                    "std": float(np.std(config_results["profits"][s])),
                }
                for s in config_results["profits"]
            },
            "ranks": {
                s: {
                    "mean": float(np.mean(config_results["ranks"][s])),
                    "std": float(np.std(config_results["ranks"][s])),
                }
                for s in config_results["ranks"]
            },
        }

        # Add PPO behavior if present
        if config_results["ppo_behavior"]:
            ppo_b = config_results["ppo_behavior"]
            all_results[config_name]["ppo_behavior"] = {
                "avg_trade_time": float(np.mean([b["avg_trade_time"] for b in ppo_b])),
                "early_trades_pct": float(np.mean([b["early_trades_pct"] for b in ppo_b])),
                "late_trades_pct": float(np.mean([b["late_trades_pct"] for b in ppo_b])),
                "avg_trade_price": float(np.mean([b["avg_trade_price"] for b in ppo_b])),
            }

    total_elapsed = time.time() - overall_start
    print(f"\nTotal time: {total_elapsed:.1f}s")

    # Print summary tables
    print("\n" + "=" * 70)
    print("MARKET-LEVEL METRICS COMPARISON")
    print("=" * 70)
    print(
        f"{'Market Type':<20} {'Efficiency':>12} {'Volatility':>12} {'V-Ineff':>10} {'Trades/P':>10}"
    )
    print("-" * 70)
    for config_name in ["zi_only", "zic_only", "zip_only", "ppo_mix"]:
        m = all_results[config_name]["market_metrics"]
        eff = f"{m['efficiency']['mean']:.1f}%"
        vol = f"{m['price_volatility']['mean']:.1f}%"
        vineff = f"{m['v_inefficiency']['mean']:.2f}"
        trades = f"{m['trades_per_period']['mean']:.1f}"
        print(f"{config_name:<20} {eff:>12} {vol:>12} {vineff:>10} {trades:>10}")

    print("\n" + "=" * 70)
    print("PROFIT BY STRATEGY (PPO+mix market)")
    print("=" * 70)
    ppo_profits = all_results["ppo_mix"]["profits"]
    sorted_profits = sorted(ppo_profits.items(), key=lambda x: -x[1]["mean"])
    for strat, data in sorted_profits:
        print(f"  {strat}: {data['mean']:,.0f} +/- {data['std']:,.0f}")

    if "ppo_behavior" in all_results["ppo_mix"]:
        print("\n" + "=" * 70)
        print("PPO TRADING BEHAVIOR")
        print("=" * 70)
        ppo_b = all_results["ppo_mix"]["ppo_behavior"]
        print(f"  Average trade time: {ppo_b['avg_trade_time']:.1f} / 100 steps")
        print(f"  Early trades (t<=30): {ppo_b['early_trades_pct']:.1f}%")
        print(f"  Late trades (t>=70): {ppo_b['late_trades_pct']:.1f}%")
        print(f"  Average trade price: {ppo_b['avg_trade_price']:.0f}")

    # Save results
    with open(results_dir / "full_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {results_dir / 'full_results.json'}")


if __name__ == "__main__":
    main()
