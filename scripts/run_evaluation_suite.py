#!/usr/bin/env python3
"""
Unified Evaluation Suite for Trading Strategies.

Three evaluation modes:
1. Self-play: All agents same strategy (market properties)
2. Zero-play: 1 focal algo vs ZIC baseline (exploitation power)
3. Mixed-play: Multi-strategy competition (direct comparison)

Usage:
    python scripts/run_evaluation_suite.py --mode self --strategy ZIC
    python scripts/run_evaluation_suite.py --mode zero --strategy Kaplan
    python scripts/run_evaluation_suite.py --mode mixed
    python scripts/run_evaluation_suite.py --mode all  # Run everything
    python scripts/run_evaluation_suite.py --summary   # Generate summary table
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator

STRATEGIES = ["ZI", "ZI2", "ZIC", "ZIP"]
MIXED_POOL = ["ZI", "ZI2", "ZIC", "ZIP"]  # 4 core strategies for mixed-play


def compute_efficiency(buyers: list, sellers: list, trade_prices: list) -> float:
    """Compute market efficiency as actual surplus / theoretical max."""
    # Actual surplus from trades
    actual_surplus = sum(b.period_profit for b in buyers) + sum(s.period_profit for s in sellers)

    # Theoretical max: sort valuations desc, costs asc, match until val < cost
    all_vals = []
    for b in buyers:
        all_vals.extend(b.valuations)
    all_costs = []
    for s in sellers:
        all_costs.extend(s.valuations)  # For sellers, valuations are costs

    all_vals.sort(reverse=True)
    all_costs.sort()

    max_surplus = 0
    for v, c in zip(all_vals, all_costs):
        if v >= c:
            max_surplus += v - c
        else:
            break

    return (actual_surplus / max_surplus * 100) if max_surplus > 0 else 0


def run_self_play(
    strategy: str,
    seed: int = 42,
    num_periods: int = 10,
    num_buyers: int = 4,
    num_sellers: int = 4,
    num_tokens: int = 4,
    num_steps: int = 100,
) -> dict:
    """Run self-play: all agents same strategy."""
    np.random.seed(seed)
    token_gen = TokenGenerator(6453, num_tokens, seed)

    efficiencies = []
    price_volatilities = []
    trade_volumes = []
    all_profits = []

    for period in range(1, num_periods + 1):
        token_gen.new_round()

        # Create all buyers with same strategy
        buyers = []
        for i in range(num_buyers):
            tokens = token_gen.generate_tokens(True)
            agent = create_agent(
                strategy,
                i + 1,
                True,
                num_tokens,
                tokens,
                seed=seed + i * 100,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
            )
            buyers.append(agent)

        # Create all sellers with same strategy
        sellers = []
        for i in range(num_sellers):
            tokens = token_gen.generate_tokens(False)
            agent = create_agent(
                strategy,
                num_buyers + i + 1,
                False,
                num_tokens,
                tokens,
                seed=seed + (num_buyers + i) * 100,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
            )
            sellers.append(agent)

        all_agents = buyers + sellers
        for agent in all_agents:
            agent.start_period(period)

        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            num_times=num_steps,
            price_min=0,
            price_max=1000,
            buyers=buyers,
            sellers=sellers,
            seed=seed + period * 10000,
        )

        # Run the period
        for step in range(1, num_steps + 1):
            market.run_time_step()

        # Collect metrics
        ob = market.orderbook
        trade_prices = [
            int(ob.trade_price[t]) for t in range(1, num_steps + 1) if ob.trade_price[t] > 0
        ]

        eff = compute_efficiency(buyers, sellers, trade_prices)
        efficiencies.append(eff)
        trade_volumes.append(len(trade_prices))

        if len(trade_prices) > 1:
            price_volatilities.append(np.std(trade_prices))
        else:
            price_volatilities.append(0)

        for b in buyers:
            all_profits.append(b.period_profit)
        for s in sellers:
            all_profits.append(s.period_profit)

    return {
        "mode": "self",
        "strategy": strategy,
        "efficiency_mean": round(np.mean(efficiencies), 1),
        "efficiency_std": round(np.std(efficiencies), 1),
        "price_volatility": round(np.mean(price_volatilities), 1),
        "trade_volume": round(np.mean(trade_volumes), 2),
        "profit_mean": round(np.mean(all_profits), 1),
        "profit_std": round(np.std(all_profits), 1),
    }


def run_zero_play(
    strategy: str,
    seed: int = 42,
    num_periods: int = 10,
    num_buyers: int = 4,
    num_sellers: int = 4,
    num_tokens: int = 4,
    num_steps: int = 100,
) -> dict:
    """Run zero-play: 1 focal buyer vs all ZIC."""
    np.random.seed(seed)
    token_gen = TokenGenerator(6453, num_tokens, seed)

    focal_profits = []
    opponent_profits = []
    trade_times = []
    spread_bid_pairs = []
    improving_bids = 0
    total_bids = 0
    first_trade_times = []
    pass_count = 0
    total_actions = 0

    for period in range(1, num_periods + 1):
        token_gen.new_round()

        # Focal buyer (strategy X)
        buyers = []
        tokens = token_gen.generate_tokens(True)
        focal = create_agent(
            strategy,
            1,
            True,
            num_tokens,
            tokens,
            seed=seed,
            num_times=num_steps,
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            price_min=0,
            price_max=1000,
        )
        buyers.append(focal)

        # Other buyers (ZIC)
        for i in range(1, num_buyers):
            tokens = token_gen.generate_tokens(True)
            agent = create_agent(
                "ZIC",
                i + 1,
                True,
                num_tokens,
                tokens,
                seed=seed + i * 100,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
            )
            buyers.append(agent)

        # All sellers (ZIC)
        sellers = []
        for i in range(num_sellers):
            tokens = token_gen.generate_tokens(False)
            agent = create_agent(
                "ZIC",
                num_buyers + i + 1,
                False,
                num_tokens,
                tokens,
                seed=seed + (num_buyers + i) * 100,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
            )
            sellers.append(agent)

        all_agents = buyers + sellers
        for agent in all_agents:
            agent.start_period(period)

        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            num_times=num_steps,
            price_min=0,
            price_max=1000,
            buyers=buyers,
            sellers=sellers,
            seed=seed + period * 10000,
        )

        first_trade_step = None

        for step in range(1, num_steps + 1):
            ob = market.orderbook
            high_bid = int(ob.high_bid[step - 1]) if step > 1 else 0
            low_ask = int(ob.low_ask[step - 1]) if step > 1 else 0
            spread = (low_ask - high_bid) if (high_bid > 0 and low_ask > 0) else 0

            market.run_time_step()

            # Track focal agent behavior
            focal_price = int(ob.bids[1, step])
            trade_price = int(ob.trade_price[step])
            focal_traded = trade_price > 0 and int(ob.high_bidder[step]) == 1

            total_actions += 1
            if focal_price <= 0:
                pass_count += 1
            else:
                total_bids += 1
                if focal_price > high_bid:
                    improving_bids += 1

            if spread > 0:
                did_bid = 1 if focal_price > 0 else 0
                spread_bid_pairs.append((spread, did_bid))

            if focal_traded:
                trade_times.append(step)
                if first_trade_step is None:
                    first_trade_step = step

        if first_trade_step:
            first_trade_times.append(first_trade_step)

        focal_profits.append(focal.period_profit)
        for b in buyers[1:]:
            opponent_profits.append(b.period_profit)
        for s in sellers:
            opponent_profits.append(s.period_profit)

    # Compute strategic metrics
    sr = 0.0
    if len(spread_bid_pairs) >= 10:
        spreads = [p[0] for p in spread_bid_pairs]
        bids = [p[1] for p in spread_bid_pairs]
        if np.std(spreads) > 0 and np.std(bids) > 0:
            sr = round(np.corrcoef(spreads, bids)[0, 1], 2)

    tpr = 0.0
    if trade_times:
        early = sum(1 for t in trade_times if t < 20) + 0.1
        late = sum(1 for t in trade_times if t >= 80)
        tpr = round(late / early, 2)

    pir = round(improving_bids / total_bids * 100, 1) if total_bids > 0 else 0
    ps = round(np.mean(first_trade_times), 1) if first_trade_times else 0
    pass_rate = round(pass_count / total_actions * 100, 1) if total_actions > 0 else 0

    focal_mean = np.mean(focal_profits)
    opp_mean = np.mean(opponent_profits) if opponent_profits else 1
    profit_ratio = round(focal_mean / opp_mean, 2) if opp_mean != 0 else 0

    return {
        "mode": "zero",
        "strategy": strategy,
        "focal_profit_mean": round(focal_mean, 1),
        "focal_profit_std": round(np.std(focal_profits), 1),
        "opponent_profit_mean": round(opp_mean, 1),
        "profit_ratio": profit_ratio,
        "total_trades": len(trade_times),
        "spread_responsiveness": sr,
        "time_pressure_response": tpr,
        "price_improvement_rate": pir,
        "patience_score": ps,
        "pass_rate": pass_rate,
    }


def run_mixed_play(
    seed: int = 42,
    num_periods: int = 10,
    num_tokens: int = 4,
    num_steps: int = 100,
) -> dict:
    """Run mixed-play: 4 strategies x 2 buyers each vs 8 ZIC sellers."""
    np.random.seed(seed)

    num_buyers = len(MIXED_POOL) * 2  # 8 buyers
    num_sellers = 8

    token_gen = TokenGenerator(6453, num_tokens, seed)

    # Track profits by strategy
    strategy_profits = {s: [] for s in MIXED_POOL}
    strategy_eq_profits = {s: [] for s in MIXED_POOL}  # Equilibrium profits for IER
    strategy_trades = {s: 0 for s in MIXED_POOL}
    strategy_wins = {s: 0 for s in MIXED_POOL}

    for period in range(1, num_periods + 1):
        token_gen.new_round()

        # Create buyers: 2 per strategy
        buyers = []
        buyer_strategies = []
        agent_id = 1
        for strat in MIXED_POOL:
            for _ in range(2):
                tokens = token_gen.generate_tokens(True)
                agent = create_agent(
                    strat,
                    agent_id,
                    True,
                    num_tokens,
                    tokens,
                    seed=seed + agent_id * 100,
                    num_times=num_steps,
                    num_buyers=num_buyers,
                    num_sellers=num_sellers,
                    price_min=0,
                    price_max=1000,
                )
                buyers.append(agent)
                buyer_strategies.append(strat)
                agent_id += 1

        # Create 8 ZIC sellers
        sellers = []
        for i in range(num_sellers):
            tokens = token_gen.generate_tokens(False)
            agent = create_agent(
                "ZIC",
                agent_id,
                False,
                num_tokens,
                tokens,
                seed=seed + agent_id * 100,
                num_times=num_steps,
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                price_min=0,
                price_max=1000,
            )
            sellers.append(agent)
            agent_id += 1

        all_agents = buyers + sellers
        for agent in all_agents:
            agent.start_period(period)

        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            num_times=num_steps,
            price_min=0,
            price_max=1000,
            buyers=buyers,
            sellers=sellers,
            seed=seed + period * 10000,
        )

        for step in range(1, num_steps + 1):
            market.run_time_step()

        # Calculate equilibrium price for IER
        all_vals = []
        for b in buyers:
            all_vals.extend(b.valuations)
        all_costs = []
        for s in sellers:
            all_costs.extend(s.valuations)  # For sellers, valuations are costs

        all_vals_sorted = sorted(all_vals, reverse=True)
        all_costs_sorted = sorted(all_costs)

        # Find Q* and P*
        q_star = 0
        for v, c in zip(all_vals_sorted, all_costs_sorted):
            if v >= c:
                q_star += 1
            else:
                break

        if q_star > 0:
            p_star = (all_vals_sorted[q_star - 1] + all_costs_sorted[q_star - 1]) / 2
        else:
            p_star = 500  # Fallback

        # Collect profits by strategy and compute equilibrium profits
        period_profits = {s: 0 for s in MIXED_POOL}
        period_eq_profits = {s: 0 for s in MIXED_POOL}
        for buyer, strat in zip(buyers, buyer_strategies):
            strategy_profits[strat].append(buyer.period_profit)
            period_profits[strat] += buyer.period_profit
            strategy_trades[strat] += buyer.num_trades

            # Equilibrium profit: for each INTRA-MARGINAL unit (v > P*)
            eq_profit = sum(v - p_star for v in buyer.valuations if v > p_star)
            strategy_eq_profits[strat].append(eq_profit)
            period_eq_profits[strat] += eq_profit

        # Who won this period?
        winner = max(period_profits.items(), key=lambda x: x[1])[0]
        strategy_wins[winner] += 1

    # Compute rankings
    mean_profits = {s: np.mean(strategy_profits[s]) for s in MIXED_POOL}
    ranked = sorted(mean_profits.items(), key=lambda x: -x[1])
    rankings = {s: i + 1 for i, (s, _) in enumerate(ranked)}

    # Compute IER = actual_profit / eq_profit
    mean_eq_profits = {s: np.mean(strategy_eq_profits[s]) for s in MIXED_POOL}
    ier = {}
    for s in MIXED_POOL:
        if mean_eq_profits[s] > 0:
            ier[s] = round(mean_profits[s] / mean_eq_profits[s], 2)
        elif mean_eq_profits[s] < 0:
            # Negative eq_profit means should have lost, flip the ratio interpretation
            ier[s] = round(mean_profits[s] / mean_eq_profits[s], 2)
        else:
            ier[s] = 0.0  # No equilibrium profit

    return {
        "mode": "mixed",
        "strategies": MIXED_POOL,
        "rankings": rankings,
        "mean_profits": {s: round(v, 1) for s, v in mean_profits.items()},
        "total_trades": strategy_trades,
        "win_counts": strategy_wins,
        "ier": ier,
        "num_periods": num_periods,
    }


def run_multi_seed(mode: str, strategy: str | None, seeds: list, num_periods: int) -> dict:
    """Run evaluation across multiple seeds."""
    results = []
    for seed in seeds:
        if mode == "self":
            r = run_self_play(strategy, seed=seed, num_periods=num_periods)
        elif mode == "zero":
            r = run_zero_play(strategy, seed=seed, num_periods=num_periods)
        elif mode == "mixed":
            r = run_mixed_play(seed=seed, num_periods=num_periods)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        results.append(r)

    # Aggregate results
    if mode in ("self", "zero"):
        aggregated = {
            "mode": mode,
            "strategy": strategy,
            "seeds": seeds,
            "num_periods": num_periods,
        }
        numeric_keys = [
            k
            for k in results[0].keys()
            if isinstance(results[0][k], (int, float)) and k not in ("mode",)
        ]
        for key in numeric_keys:
            vals = [r[key] for r in results]
            aggregated[key] = round(np.mean(vals), 2)
            aggregated[f"{key}_std"] = round(np.std(vals), 2)
        return aggregated
    else:  # mixed
        aggregated = {
            "mode": "mixed",
            "strategies": MIXED_POOL,
            "seeds": seeds,
            "num_periods": num_periods,
        }
        # Average rankings
        avg_rankings = {s: np.mean([r["rankings"][s] for r in results]) for s in MIXED_POOL}
        aggregated["avg_rankings"] = {s: round(v, 2) for s, v in avg_rankings.items()}
        # Average profits
        avg_profits = {s: np.mean([r["mean_profits"][s] for r in results]) for s in MIXED_POOL}
        aggregated["avg_profits"] = {s: round(v, 1) for s, v in avg_profits.items()}
        # Total trades
        total_trades = {s: sum(r["total_trades"][s] for r in results) for s in MIXED_POOL}
        aggregated["total_trades"] = total_trades
        # Win rates
        total_wins = {s: sum(r["win_counts"][s] for r in results) for s in MIXED_POOL}
        total_periods = len(seeds) * num_periods
        aggregated["win_rates"] = {
            s: round(w / total_periods * 100, 1) for s, w in total_wins.items()
        }
        # Average IER
        avg_ier = {s: np.mean([r["ier"][s] for r in results]) for s in MIXED_POOL}
        aggregated["avg_ier"] = {s: round(v, 2) for s, v in avg_ier.items()}
        return aggregated


def print_results(results: dict):
    """Print results in formatted output."""
    mode = results.get("mode", "unknown")
    print(f"\n{'='*60}")
    print(f"MODE: {mode.upper()}")
    print(f"{'='*60}")

    if mode == "self":
        print(f"Strategy: {results['strategy']}")
        print(
            f"Efficiency: {results.get('efficiency_mean', 'N/A')}% (+/- {results.get('efficiency_mean_std', 0)})"
        )
        print(f"Price Volatility: {results.get('price_volatility', 'N/A')}")
        print(f"Trade Volume: {results.get('trade_volume', 'N/A')} per period")
        print(f"Profit: {results.get('profit_mean', 'N/A')} (+/- {results.get('profit_std', 0)})")

    elif mode == "zero":
        print(f"Strategy: {results['strategy']}")
        print(f"Focal Profit: {results.get('focal_profit_mean', 'N/A')}")
        print(f"Opponent Profit: {results.get('opponent_profit_mean', 'N/A')}")
        print(f"Profit Ratio: {results.get('profit_ratio', 'N/A')}x")
        print("\nStrategic Metrics:")
        print(f"  SR (Spread Responsiveness): {results.get('spread_responsiveness', 0)}")
        print(f"  TPR (Time Pressure Response): {results.get('time_pressure_response', 0)}")
        print(f"  PIR (Price Improvement Rate): {results.get('price_improvement_rate', 0)}%")
        print(f"  PS (Patience Score): {results.get('patience_score', 0)}")
        print(f"  PASS Rate: {results.get('pass_rate', 0)}%")

    elif mode == "mixed":
        print(f"Strategies: {results['strategies']}")
        print("\nRankings (1=best):")
        for s, r in sorted(results.get("avg_rankings", {}).items(), key=lambda x: x[1]):
            profit = results.get("avg_profits", {}).get(s, 0)
            win_rate = results.get("win_rates", {}).get(s, 0)
            print(f"  #{r:.1f} {s}: profit={profit}, win_rate={win_rate}%")


def generate_summary():
    """Generate summary table from saved results."""
    results_dir = Path("results")

    # Load all eval results
    self_results = {}
    zero_results = {}

    for f in results_dir.glob("eval_self_*.json"):
        strat = f.stem.replace("eval_self_", "")
        with open(f) as fp:
            self_results[strat] = json.load(fp)

    for f in results_dir.glob("eval_zero_*.json"):
        strat = f.stem.replace("eval_zero_", "")
        with open(f) as fp:
            zero_results[strat] = json.load(fp)

    mixed_path = results_dir / "eval_mixed.json"
    mixed_results = {}
    if mixed_path.exists():
        with open(mixed_path) as fp:
            mixed_results = json.load(fp)

    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    print("\n| Strategy | Self-Eff | Zero-Ratio | Mixed-Rank | SR | TPR | PIR | PS | PASS% |")
    print("|----------|----------|------------|------------|-----|-----|-----|-----|-------|")

    for strat in STRATEGIES:
        self_eff = self_results.get(strat, {}).get("efficiency_mean", "N/A")
        zero_ratio = zero_results.get(strat, {}).get("profit_ratio", "N/A")
        mixed_rank = mixed_results.get("avg_rankings", {}).get(strat, "N/A")
        sr = zero_results.get(strat, {}).get("spread_responsiveness", "N/A")
        tpr = zero_results.get(strat, {}).get("time_pressure_response", "N/A")
        pir = zero_results.get(strat, {}).get("price_improvement_rate", "N/A")
        ps = zero_results.get(strat, {}).get("patience_score", "N/A")
        pass_r = zero_results.get(strat, {}).get("pass_rate", "N/A")

        self_str = f"{self_eff}%" if self_eff != "N/A" else "N/A"
        zero_str = f"{zero_ratio}x" if zero_ratio != "N/A" else "N/A"
        mixed_str = f"#{mixed_rank}" if mixed_rank != "N/A" else "N/A"

        print(
            f"| {strat:8} | {self_str:>8} | {zero_str:>10} | {mixed_str:>10} | {sr:>4} | {tpr:>4} | {pir:>4} | {ps:>4} | {pass_r:>5} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite")
    parser.add_argument(
        "--mode", type=str, choices=["self", "zero", "mixed", "all"], help="Evaluation mode"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=STRATEGIES,
        help="Strategy to evaluate (for self/zero modes)",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--periods", type=int, default=10, help="Periods per seed")
    parser.add_argument("--summary", action="store_true", help="Generate summary table")
    parser.add_argument("--output", type=str, help="Output file path")
    args = parser.parse_args()

    seeds = [42, 123, 456, 789, 1000][: args.seeds]

    if args.summary:
        generate_summary()
        return

    if args.mode == "all":
        # Run all modes for all strategies
        for mode in ["self", "zero"]:
            for strat in STRATEGIES:
                print(f"Running {mode} for {strat}...")
                results = run_multi_seed(mode, strat, seeds, args.periods)
                output = Path(f"results/eval_{mode}_{strat}.json")
                with open(output, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"  Saved to {output}")

        print("Running mixed...")
        results = run_multi_seed("mixed", None, seeds, args.periods)
        output = Path("results/eval_mixed.json")
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved to {output}")

        generate_summary()
        return

    if args.mode in ("self", "zero") and not args.strategy:
        # Run for all strategies
        for strat in STRATEGIES:
            print(f"Running {args.mode} for {strat}...")
            results = run_multi_seed(args.mode, strat, seeds, args.periods)
            output = Path(args.output or f"results/eval_{args.mode}_{strat}.json")
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            print_results(results)
    elif args.mode == "mixed":
        results = run_multi_seed("mixed", None, seeds, args.periods)
        output = Path(args.output or "results/eval_mixed.json")
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print_results(results)
    elif args.mode and args.strategy:
        results = run_multi_seed(args.mode, args.strategy, seeds, args.periods)
        output = Path(args.output or f"results/eval_{args.mode}_{args.strategy}.json")
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        print_results(results)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
