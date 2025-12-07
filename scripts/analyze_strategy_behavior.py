#!/usr/bin/env python3
"""
Unified Behavioral Analysis for Any Trading Strategy.

Generates standardized behavioral metrics that are directly comparable
across strategies (ZI, ZIC, ZIP, GD, Kaplan, Skeleton, PPO, LLM).

Usage:
    python scripts/analyze_strategy_behavior.py --strategy ZIC
    python scripts/analyze_strategy_behavior.py --strategy GD --seeds 5 --periods 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
from collections import defaultdict

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator

SUPPORTED_STRATEGIES = ["ZI", "ZI2", "ZIC", "ZIP", "GD", "Kaplan", "Skeleton", "BGAN", "Staecker"]


def run_traced_market(
    strategy: str, seed: int = 42, num_periods: int = 3, verbose: bool = True, mode: str = "zero"
):
    """Run a market with focal strategy and trace all decisions.

    Args:
        mode: "zero" = 1 focal buyer vs ZIC, "self" = all agents same strategy
    """

    np.random.seed(seed)

    num_buyers = 4
    num_sellers = 4
    num_tokens = 4
    num_steps = 100
    gametype = 6453

    token_gen = TokenGenerator(gametype, num_tokens, seed)

    all_traces = []
    action_counts = defaultdict(int)
    trade_timing = []
    shade_values = []
    total_profit = 0
    total_trades = 0

    # Strategic metrics tracking
    spread_bid_pairs = []  # (spread, did_bid) for spread responsiveness
    improving_bids = 0
    total_bids = 0
    first_trade_times = []  # Time of first trade in each period
    eq_profit_total = 0  # Equilibrium profit for SCR

    for period in range(1, num_periods + 1):
        token_gen.new_round()

        # Create agents based on mode
        buyers = []
        for i in range(num_buyers):
            tokens = token_gen.generate_tokens(True)
            # In self mode, all agents use same strategy; in zero mode, only first buyer
            use_strategy = strategy if (mode == "self" or i == 0) else "ZIC"
            agent = create_agent(
                use_strategy,
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

        sellers = []
        for i in range(num_sellers):
            tokens = token_gen.generate_tokens(False)
            # In self mode, sellers also use same strategy
            use_strategy = strategy if mode == "self" else "ZIC"
            agent = create_agent(
                use_strategy,
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

        focal_agent = buyers[0]

        if verbose:
            print(f"\n{'='*60}")
            print(f"PERIOD {period}")
            print(f"{strategy} Valuations: {focal_agent.valuations}")
            print(f"{'='*60}")

        period_traces = []

        for step in range(1, num_steps + 1):
            # Before step - capture state
            ob = market.orderbook
            high_bid = int(ob.high_bid[step - 1]) if step > 1 else 0
            low_ask = int(ob.low_ask[step - 1]) if step > 1 else 0
            spread = (low_ask - high_bid) if (high_bid > 0 and low_ask > 0) else 0
            spread_pct = spread / low_ask * 100 if low_ask > 0 else 100

            focal_val = focal_agent.get_current_valuation()
            focal_tokens_left = focal_agent.num_tokens - focal_agent.num_trades

            # Run step
            market.run_time_step()

            # After step - get focal agent's action
            focal_price = int(ob.bids[1, step])  # Focal agent is buyer 1
            trade_price = int(ob.trade_price[step])
            focal_traded = trade_price > 0 and int(ob.high_bidder[step]) == 1

            # Deduce action type
            action_type = "UNKNOWN"
            is_sniper_window = step >= num_steps - 2  # Final 2 steps
            is_spread_narrow = spread_pct < 10 if spread > 0 else False

            if focal_price <= 0:
                action_type = "PASS"
            elif focal_traded:
                if is_sniper_window:
                    action_type = "SNIPE/TRADE"
                elif is_spread_narrow:
                    action_type = "JUMP/TRADE"
                else:
                    action_type = "ACCEPT/TRADE"
                trade_timing.append(step)
                profit = focal_val - trade_price
                total_profit += profit
                total_trades += 1
            else:
                # Bid but no trade - classify by context
                if is_sniper_window:
                    action_type = "SNIPE_BID"
                elif is_spread_narrow:
                    action_type = "JUMP_BID"
                else:
                    # Calculate shade percentage
                    if focal_val > 0 and focal_price > 0:
                        shade = (focal_val - focal_price) / focal_val * 100
                        if shade > 0:
                            shade_values.append(shade)
                            action_type = "Shade"
                        elif shade < -5:  # Bidding above value
                            action_type = "ABOVE_VALUE"
                        else:
                            action_type = "TRUTHFUL"
                    else:
                        action_type = "BID"

            # Count action types (simplified)
            if action_type == "PASS":
                action_counts["PASS"] += 1
            elif "SNIPE" in action_type:
                action_counts["SNIPE"] += 1
            elif "JUMP" in action_type:
                action_counts["JUMP"] += 1
            elif "ACCEPT" in action_type or "TRADE" in action_type:
                action_counts["ACCEPT/TRADE"] += 1
            elif "Shade" in action_type:
                action_counts["Shade"] += 1
            elif "TRUTHFUL" in action_type:
                action_counts["TRUTHFUL"] += 1
            elif "ABOVE_VALUE" in action_type:
                action_counts["ABOVE_VALUE"] += 1
            else:
                action_counts["OTHER"] += 1

            # Track strategic metrics
            # 1. Spread Responsiveness: track (spread, did_bid) pairs
            if spread > 0:  # Only when spread is defined
                did_bid = 1 if focal_price > 0 else 0
                spread_bid_pairs.append((spread, did_bid))

            # 2. Price Improvement Rate: did this bid improve the best?
            if focal_price > 0:
                total_bids += 1
                if focal_price > high_bid:
                    improving_bids += 1

            # 3. First trade time tracking (handled at period level below)

            trace = {
                "step": step,
                "high_bid": high_bid,
                "low_ask": low_ask,
                "spread": spread,
                "spread_pct": spread_pct,
                "focal_val": focal_val,
                "focal_tokens": focal_tokens_left,
                "focal_price": focal_price,
                "action_type": action_type,
                "traded": focal_traded,
                "trade_price": trade_price if focal_traded else 0,
            }
            period_traces.append(trace)

            if verbose and (focal_price > 0 or focal_traded):
                profit_str = (
                    f" -> TRADE@{trade_price} profit={focal_val - trade_price}"
                    if focal_traded
                    else ""
                )
                print(
                    f"t={step:3d} | bid={high_bid:4d} ask={low_ask:4d} | "
                    f"val={focal_val:4d} | {strategy}={focal_price:4d} [{action_type:15s}]{profit_str}"
                )

        all_traces.append(period_traces)

        # Track first trade time for this period
        period_trade_times = [t["step"] for t in period_traces if t["traded"]]
        if period_trade_times:
            first_trade_times.append(min(period_trade_times))

        if verbose:
            print(
                f"\nPeriod {period} Summary: {strategy} profit={focal_agent.period_profit}, trades={focal_agent.num_trades}"
            )

    # Build strategic metrics dict
    strategic_metrics = {
        "spread_bid_pairs": spread_bid_pairs,
        "improving_bids": improving_bids,
        "total_bids": total_bids,
        "first_trade_times": first_trade_times,
    }

    return (
        all_traces,
        action_counts,
        trade_timing,
        shade_values,
        total_profit,
        total_trades,
        strategic_metrics,
    )


def run_multi_seed_analysis(
    strategy: str, seeds: list, num_periods: int = 5, verbose: bool = False, mode: str = "zero"
):
    """Run analysis across multiple seeds.

    Args:
        mode: "zero" = 1 focal buyer vs ZIC, "self" = all agents same strategy
    """
    all_action_counts = defaultdict(int)
    all_trade_timing = []
    all_shade_values = []
    total_profit = 0
    total_trades = 0

    # Aggregate strategic metrics
    all_spread_bid_pairs = []
    all_improving_bids = 0
    all_total_bids = 0
    all_first_trade_times = []

    for seed in seeds:
        (
            traces,
            action_counts,
            trade_timing,
            shade_values,
            profit,
            trades,
            strategic_metrics,
        ) = run_traced_market(
            strategy=strategy, seed=seed, num_periods=num_periods, verbose=verbose, mode=mode
        )

        for k, v in action_counts.items():
            all_action_counts[k] += v
        all_trade_timing.extend(trade_timing)
        all_shade_values.extend(shade_values)
        total_profit += profit
        total_trades += trades

        # Aggregate strategic metrics
        all_spread_bid_pairs.extend(strategic_metrics["spread_bid_pairs"])
        all_improving_bids += strategic_metrics["improving_bids"]
        all_total_bids += strategic_metrics["total_bids"]
        all_first_trade_times.extend(strategic_metrics["first_trade_times"])

    aggregated_strategic = {
        "spread_bid_pairs": all_spread_bid_pairs,
        "improving_bids": all_improving_bids,
        "total_bids": all_total_bids,
        "first_trade_times": all_first_trade_times,
    }

    return (
        all_action_counts,
        all_trade_timing,
        all_shade_values,
        total_profit,
        total_trades,
        aggregated_strategic,
    )


def compute_metrics(
    action_counts, trade_timing, shade_values, total_profit, total_trades, strategic_metrics=None
):
    """Compute standardized metrics dict."""
    total_actions = sum(action_counts.values())

    # Action distribution
    action_dist = {}
    for action, count in action_counts.items():
        action_dist[action] = round(count / total_actions * 100, 1) if total_actions > 0 else 0

    # Trade timing
    if trade_timing:
        mean_trade_time = round(np.mean(trade_timing), 1)
        std_trade_time = round(np.std(trade_timing), 1)
        n = len(trade_timing)
        early_pct = round(sum(1 for t in trade_timing if t < 30) / n * 100, 1)
        mid_pct = round(sum(1 for t in trade_timing if 30 <= t < 70) / n * 100, 1)
        late_pct = round(sum(1 for t in trade_timing if t >= 70) / n * 100, 1)
    else:
        mean_trade_time = std_trade_time = 0
        early_pct = mid_pct = late_pct = 0

    # Shade analysis
    if shade_values:
        mean_shade = round(np.mean(shade_values), 1)
        std_shade = round(np.std(shade_values), 1)
    else:
        mean_shade = std_shade = 0

    # Profit
    profit_per_trade = round(total_profit / total_trades, 1) if total_trades > 0 else 0

    # Dominant action
    dominant_action = max(action_dist.items(), key=lambda x: x[1])[0] if action_dist else "NONE"
    dominant_pct = max(action_dist.values()) if action_dist else 0

    # Strategic metrics
    spread_responsiveness = 0.0
    time_pressure_response = 0.0
    price_improvement_rate = 0.0
    patience_score = 0.0
    pass_rate = action_dist.get("PASS", 0.0)

    if strategic_metrics:
        # Spread Responsiveness: Corr(spread, bid_activity)
        # Negative = bids when spread is narrow (strategic)
        spread_bid_pairs = strategic_metrics.get("spread_bid_pairs", [])
        if len(spread_bid_pairs) >= 10:
            spreads = [p[0] for p in spread_bid_pairs]
            bids = [p[1] for p in spread_bid_pairs]
            if np.std(spreads) > 0 and np.std(bids) > 0:
                spread_responsiveness = round(np.corrcoef(spreads, bids)[0, 1], 2)

        # Time Pressure Response: trades_late / trades_early
        # TPR > 1 = sniper, TPR < 1 = eager, TPR ~1 = uniform
        if trade_timing:
            trades_early = sum(1 for t in trade_timing if t < 20) + 0.1  # Avoid div/0
            trades_late = sum(1 for t in trade_timing if t >= 80)
            time_pressure_response = round(trades_late / trades_early, 2)

        # Price Improvement Rate: improving_bids / total_bids
        total_bids = strategic_metrics.get("total_bids", 0)
        improving_bids = strategic_metrics.get("improving_bids", 0)
        if total_bids > 0:
            price_improvement_rate = round(improving_bids / total_bids * 100, 1)

        # Patience Score: mean time to first trade
        first_trade_times = strategic_metrics.get("first_trade_times", [])
        if first_trade_times:
            patience_score = round(np.mean(first_trade_times), 1)

    return {
        "action_distribution": action_dist,
        "dominant_action": f"{dominant_action} ({dominant_pct}%)",
        "mean_trade_time": mean_trade_time,
        "std_trade_time": std_trade_time,
        "early_pct": early_pct,
        "mid_pct": mid_pct,
        "late_pct": late_pct,
        "mean_shade_pct": mean_shade,
        "std_shade_pct": std_shade,
        "total_profit": total_profit,
        "total_trades": total_trades,
        "profit_per_trade": profit_per_trade,
        # Strategic metrics
        "spread_responsiveness": spread_responsiveness,
        "time_pressure_response": time_pressure_response,
        "price_improvement_rate": price_improvement_rate,
        "patience_score": patience_score,
        "pass_rate": pass_rate,
    }


def print_analysis_results(strategy: str, metrics: dict):
    """Print analysis results in standard format."""
    print(f"\n{'='*70}")
    print(f"{strategy.upper()} TRADING BEHAVIOR ANALYSIS")
    print(f"{'='*70}")

    print("\nACTION DISTRIBUTION:")
    for action, pct in sorted(metrics["action_distribution"].items(), key=lambda x: -x[1]):
        print(f"  {action:20s}: {pct:5.1f}%")

    print(f"\nDOMINANT ACTION: {metrics['dominant_action']}")

    print("\nTRADE TIMING:")
    print(f"  Total trades: {metrics['total_trades']}")
    print(f"  Mean time: {metrics['mean_trade_time']} +/- {metrics['std_trade_time']}")
    print(f"  Early (t<30):  {metrics['early_pct']:.1f}%")
    print(f"  Mid (30-70):   {metrics['mid_pct']:.1f}%")
    print(f"  Late (t>=70):  {metrics['late_pct']:.1f}%")

    print("\nSHADE ANALYSIS:")
    print(f"  Mean shade: {metrics['mean_shade_pct']}%")
    print(f"  Std shade: {metrics['std_shade_pct']}%")

    print("\nPROFIT ANALYSIS:")
    print(f"  Total profit: {metrics['total_profit']}")
    print(f"  Profit per trade: {metrics['profit_per_trade']}")

    print("\nSTRATEGIC METRICS:")
    print(f"  Spread Responsiveness (SR): {metrics.get('spread_responsiveness', 0)}")
    print(f"  Time Pressure Response (TPR): {metrics.get('time_pressure_response', 0)}")
    print(f"  Price Improvement Rate (PIR): {metrics.get('price_improvement_rate', 0)}%")
    print(f"  Patience Score (PS): {metrics.get('patience_score', 0)}")
    print(f"  PASS Rate: {metrics.get('pass_rate', 0)}%")

    # Interpretation
    sr = metrics.get("spread_responsiveness", 0)
    tpr = metrics.get("time_pressure_response", 0)
    pir = metrics.get("price_improvement_rate", 0)
    print("\nINTERPRETATION:")
    if sr < -0.3:
        print("  - SR < -0.3: Strategic (bids when spread narrows)")
    elif sr > 0.3:
        print("  - SR > 0.3: Counter-strategic (bids when spread widens)")
    else:
        print("  - SR ~0: State-independent (random/mechanical)")
    if tpr > 2:
        print("  - TPR > 2: Sniper (trades late)")
    elif tpr < 0.5:
        print("  - TPR < 0.5: Eager (trades early)")
    else:
        print("  - TPR ~1: Uniform timing")
    if pir > 70:
        print("  - PIR > 70%: Efficient bidder (most bids improve)")
    elif pir < 30:
        print("  - PIR < 30%: Inefficient bidder (wasted bids)")

    # Print table row format for behavior.md
    print(f"\n{'='*70}")
    print("TABLE ROW (for behavior.md):")
    print(
        f"| {strategy} | {metrics['dominant_action']} | {metrics['mean_trade_time']} | "
        f"{metrics['early_pct']}% | {metrics['mid_pct']}% | {metrics['late_pct']}% | "
        f"{metrics['profit_per_trade']} |"
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze trading strategy behavior")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=SUPPORTED_STRATEGIES,
        help=f"Strategy to analyze: {SUPPORTED_STRATEGIES}",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--periods", type=int, default=5, help="Periods per seed")
    parser.add_argument("--verbose", action="store_true", help="Show detailed traces")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    parser.add_argument(
        "--output", type=str, help="Save JSON to file (e.g., results/behavior_ZIC.json)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="zero",
        choices=["zero", "self"],
        help="Mode: zero = 1 focal vs ZIC, self = all agents same strategy",
    )
    args = parser.parse_args()

    seeds = [42, 123, 456, 789, 1000][: args.seeds]

    print("=" * 70)
    print(f"{args.strategy.upper()} TRADING BEHAVIOR ANALYSIS")
    print("=" * 70)
    print(
        f"\nConfig: {len(seeds)} seeds x {args.periods} periods = {len(seeds)*args.periods} periods total"
    )
    print(f"Seeds: {seeds}")
    if args.mode == "self":
        print(f"Market: 4 {args.strategy} buyers vs 4 {args.strategy} sellers (self-play)")
    else:
        print(f"Market: 1 {args.strategy} buyer + 3 ZIC buyers vs 4 ZIC sellers (zero-play)")

    (
        action_counts,
        trade_timing,
        shade_values,
        total_profit,
        total_trades,
        strategic_metrics,
    ) = run_multi_seed_analysis(
        strategy=args.strategy,
        seeds=seeds,
        num_periods=args.periods,
        verbose=args.verbose,
        mode=args.mode,
    )

    metrics = compute_metrics(
        action_counts, trade_timing, shade_values, total_profit, total_trades, strategic_metrics
    )
    metrics["strategy"] = args.strategy
    metrics["mode"] = args.mode
    metrics["config"] = {"seeds": seeds, "periods": args.periods, "mode": args.mode}

    if args.output:
        # Save to file
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {args.output}")
    elif args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print_analysis_results(args.strategy, metrics)


if __name__ == "__main__":
    main()
