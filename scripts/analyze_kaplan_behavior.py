#!/usr/bin/env python3
"""
Analyze Kaplan Trading Behavior - Trace actual decisions to understand strategy.

This script generates quantitative behavioral metrics for Kaplan that are
directly comparable to the PPO metrics from analyze_ppo_behavior.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import defaultdict

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator


def run_traced_market(seed=42, num_periods=3, verbose=True):
    """Run a market with Kaplan and trace all decisions."""

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

    for period in range(1, num_periods + 1):
        token_gen.new_round()

        # Create agents - Kaplan as buyer #1, rest ZIC
        buyers = []
        for i in range(num_buyers):
            tokens = token_gen.generate_tokens(True)
            if i == 0:  # Kaplan buyer
                agent = create_agent(
                    "Kaplan",
                    i + 1,
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
            else:  # ZIC buyers
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

        kaplan = buyers[0]

        if verbose:
            print(f"\n{'='*60}")
            print(f"PERIOD {period}")
            print(f"Kaplan Valuations: {kaplan.valuations}")
            print(f"{'='*60}")

        period_traces = []

        for step in range(1, num_steps + 1):
            # Before step - capture state
            ob = market.orderbook
            high_bid = int(ob.high_bid[step - 1]) if step > 1 else 0
            low_ask = int(ob.low_ask[step - 1]) if step > 1 else 0
            spread = (low_ask - high_bid) if (high_bid > 0 and low_ask > 0) else 0
            spread_pct = spread / low_ask * 100 if low_ask > 0 else 100

            kaplan_val = kaplan.get_current_valuation()
            kaplan_tokens_left = kaplan.num_tokens - kaplan.num_trades

            # Run step
            market.run_time_step()

            # After step - get Kaplan's action
            kaplan_price = int(ob.bids[1, step])  # Kaplan is buyer 1
            trade_price = int(ob.trade_price[step])
            kaplan_traded = trade_price > 0 and int(ob.high_bidder[step]) == 1

            # Deduce action type
            action_type = "UNKNOWN"
            is_sniper_window = step >= num_steps - 2  # Final 2 steps
            is_spread_narrow = spread_pct < 10 if spread > 0 else False

            if kaplan_price <= 0:
                action_type = "PASS"
            elif kaplan_traded:
                if is_sniper_window:
                    action_type = "SNIPE/TRADE"
                elif is_spread_narrow:
                    action_type = "JUMP/TRADE"
                else:
                    action_type = "ACCEPT/TRADE"
                trade_timing.append(step)
            else:
                # Bid but no trade - classify by context
                if is_sniper_window:
                    action_type = "SNIPE_BID"
                elif is_spread_narrow:
                    action_type = "JUMP_BID"
                else:
                    # Calculate shade percentage
                    if kaplan_val > 0 and kaplan_price > 0:
                        shade = (kaplan_val - kaplan_price) / kaplan_val * 100
                        shade_values.append(shade)
                        action_type = f"Shade ~{shade:.0f}%"
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
            else:
                action_counts["OTHER"] += 1

            trace = {
                "step": step,
                "high_bid": high_bid,
                "low_ask": low_ask,
                "spread": spread,
                "spread_pct": spread_pct,
                "kaplan_val": kaplan_val,
                "kaplan_tokens": kaplan_tokens_left,
                "kaplan_price": kaplan_price,
                "action_type": action_type,
                "traded": kaplan_traded,
                "trade_price": trade_price if kaplan_traded else 0,
                "is_sniper_window": is_sniper_window,
                "is_spread_narrow": is_spread_narrow,
            }
            period_traces.append(trace)

            if verbose and (kaplan_price > 0 or kaplan_traded):
                profit = kaplan_val - trade_price if kaplan_traded else 0
                print(
                    f"t={step:3d} | bid={high_bid:4d} ask={low_ask:4d} spread={spread:4d} ({spread_pct:4.1f}%) | "
                    f"val={kaplan_val:4d} tokens={kaplan_tokens_left} | "
                    f"Kaplan={kaplan_price:4d} [{action_type:15s}]"
                    + (f" -> TRADE@{trade_price} profit={profit}" if kaplan_traded else "")
                )

        all_traces.append(period_traces)

        if verbose:
            print(
                f"\nPeriod {period} Summary: Kaplan profit={kaplan.period_profit}, trades={kaplan.num_trades}"
            )

    return all_traces, action_counts, trade_timing, shade_values


def run_multi_seed_analysis(seeds, num_periods=5, verbose=False):
    """Run analysis across multiple seeds."""
    all_action_counts = defaultdict(int)
    all_trade_timing = []
    all_shade_values = []
    total_profit = 0
    total_trades = 0

    for seed in seeds:
        traces, action_counts, trade_timing, shade_values = run_traced_market(
            seed=seed, num_periods=num_periods, verbose=verbose
        )

        for k, v in action_counts.items():
            all_action_counts[k] += v
        all_trade_timing.extend(trade_timing)
        all_shade_values.extend(shade_values)

        # Calculate profit from traces
        for period_traces in traces:
            for t in period_traces:
                if t.get("traded"):
                    profit = t.get("kaplan_val", 0) - t.get("trade_price", 0)
                    total_profit += profit
                    total_trades += 1

    return all_action_counts, all_trade_timing, all_shade_values, total_profit, total_trades


def print_analysis_results(action_counts, trade_timing, shade_values, total_profit, total_trades):
    """Print analysis results."""
    print(f"\n{'='*70}")
    print("KAPLAN TRADING BEHAVIOR ANALYSIS")
    print(f"{'='*70}")

    print("\nACTION DISTRIBUTION:")
    total = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {action:20s}: {count:5d} ({count/total*100:5.1f}%)")

    print("\nTRADE TIMING:")
    if trade_timing:
        print(f"  Total trades: {len(trade_timing)}")
        print(f"  Mean time: {np.mean(trade_timing):.1f} +/- {np.std(trade_timing):.1f}")
        early = sum(1 for t in trade_timing if t < 30)
        mid = sum(1 for t in trade_timing if 30 <= t < 70)
        late = sum(1 for t in trade_timing if t >= 70)
        n = len(trade_timing)
        print(f"  Early (t<30):  {early:3d} ({early/n*100:5.1f}%)")
        print(f"  Mid (30-70):   {mid:3d} ({mid/n*100:5.1f}%)")
        print(f"  Late (t>=70):  {late:3d} ({late/n*100:5.1f}%)")
    else:
        print("  No trades recorded!")

    print("\nSHADE ANALYSIS:")
    if shade_values:
        print(f"  Mean shade: {np.mean(shade_values):.1f}%")
        print(f"  Std shade: {np.std(shade_values):.1f}%")
        bins = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        print("  Distribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for s in shade_values if bins[i] <= s < bins[i + 1])
            if count > 0:
                print(
                    f"    {bins[i]:2d}-{bins[i+1]:2d}%: {count:4d} ({count/len(shade_values)*100:5.1f}%)"
                )
    else:
        print("  No shade values recorded!")

    print("\nPROFIT ANALYSIS:")
    print(f"  Total profit: {total_profit}")
    print(f"  Total trades: {total_trades}")
    if total_trades > 0:
        print(f"  Profit per trade: {total_profit/total_trades:.1f}")


def print_comparison_with_ppo():
    """Print PPO comparison data for reference."""
    print(f"\n{'='*70}")
    print("PPO COMPARISON DATA (from analyze_ppo_behavior.py)")
    print(f"{'='*70}")
    print(
        """
PPO Behavioral Metrics (5 seeds x 5 periods, vs ZIC):
- Action distribution: 92.1% Shade, 3.6% Pass, 2.3% Accept
- Mean trade time: 7.8 +/- 7.1 steps
- Early trades (t<30): 98.2%
- Mid trades (30-70): 1.8%
- Late trades (t>=70): 0.0%
- Mean shade: 22.5% +/- 12.0%
- Shade at 30-40%: 66.8%
- Shade at 0-5%: 20.3%
"""
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--periods", type=int, default=5, help="Periods per seed")
    parser.add_argument("--verbose", action="store_true", help="Show detailed traces")
    parser.add_argument("--compare", action="store_true", help="Show PPO comparison")
    args = parser.parse_args()

    print("=" * 70)
    print("KAPLAN TRADING BEHAVIOR ANALYSIS")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 1000][: args.seeds]
    num_periods = args.periods

    print(
        f"\nConfig: {len(seeds)} seeds x {num_periods} periods = {len(seeds)*num_periods} periods total"
    )
    print(f"Seeds: {seeds}")
    print("Market: 1 Kaplan buyer + 3 ZIC buyers vs 4 ZIC sellers")

    action_counts, trade_timing, shade_values, total_profit, total_trades = run_multi_seed_analysis(
        seeds, num_periods=num_periods, verbose=args.verbose
    )

    print_analysis_results(action_counts, trade_timing, shade_values, total_profit, total_trades)

    if args.compare:
        print_comparison_with_ppo()


if __name__ == "__main__":
    main()
