#!/usr/bin/env python3
"""
Analyze PPO Trading Behavior - Trace actual decisions to understand strategy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import defaultdict

import numpy as np

from engine.agent_factory import create_agent
from engine.market import Market
from engine.token_generator import TokenGenerator

# Action labels for PPO's 24-action space
ACTION_LABELS = {
    0: "PASS",
    1: "ACCEPT",
    2: "Improve 0.5%",
    3: "Improve 1%",
    4: "Improve 2%",
    5: "Improve 5%",
    6: "Improve 10%",
    7: "Improve 15%",
    8: "Improve 25%",
    9: "Improve 40%",
    10: "Shade 1%",
    11: "Shade 3%",
    12: "Shade 5%",
    13: "Shade 10%",
    14: "Shade 15%",
    15: "Shade 20%",
    16: "Shade 30%",
    17: "Shade 40%",
    18: "TRUTHFUL",
    19: "JUMP_BEST",
    20: "SNIPE",
    21: "Undercut 2",
    22: "Undercut 5",
    23: "Undercut 10",
}


def run_traced_market(seed=42, num_periods=3, verbose=True, opponent_type="zic", model_path=None):
    """Run a market with PPO and trace all decisions.

    Args:
        opponent_type: "zic" for all ZIC, "mixed" for Skeleton/ZIP/Kaplan mix
        model_path: Path to PPO model checkpoint
    """

    np.random.seed(seed)

    num_buyers = 4
    num_sellers = 4
    num_tokens = 4
    num_steps = 100
    gametype = 6453

    token_gen = TokenGenerator(gametype, num_tokens, seed)

    # Create PPO as buyer #1
    ppo_model = model_path or "checkpoints/ppo_v10_10M/ppo_double_auction_8000000_steps.zip"

    # Define opponent types for mixed mode
    if opponent_type == "mixed":
        buyer_types = ["Skeleton", "ZIP", "Kaplan"]  # 3 other buyers
        seller_types = ["Skeleton", "ZIP", "Kaplan", "ZIC"]  # 4 sellers
    else:
        buyer_types = ["ZIC", "ZIC", "ZIC"]
        seller_types = ["ZIC", "ZIC", "ZIC", "ZIC"]

    all_traces = []
    action_counts = defaultdict(int)
    trade_timing = []

    for period in range(1, num_periods + 1):
        token_gen.new_round()

        # Create agents
        buyers = []
        for i in range(num_buyers):
            tokens = token_gen.generate_tokens(True)
            if i == 0:  # PPO buyer
                agent = create_agent(
                    "PPO",
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
                    model_path=ppo_model,
                )
            else:  # Other buyers based on opponent_type
                agent_type = buyer_types[i - 1]
                agent = create_agent(
                    agent_type,
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
            agent_type = seller_types[i]
            agent = create_agent(
                agent_type,
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

        ppo = buyers[0]

        if verbose:
            print(f"\n{'='*60}")
            print(f"PERIOD {period}")
            print(f"PPO Valuations: {ppo.valuations}")
            print(f"{'='*60}")

        period_traces = []

        for step in range(1, num_steps + 1):
            # Before step - capture state
            ob = market.orderbook
            high_bid = int(ob.high_bid[step - 1]) if step > 1 else 0
            low_ask = int(ob.low_ask[step - 1]) if step > 1 else 0
            spread = (low_ask - high_bid) if (high_bid > 0 and low_ask > 0) else 0

            ppo_val = ppo.get_current_valuation()
            ppo_tokens_left = ppo.num_tokens - ppo.num_trades

            # Run step
            market.run_time_step()

            # After step - get PPO's action
            ppo_price = int(ob.bids[1, step])  # PPO is buyer 1
            trade_price = int(ob.trade_price[step])
            ppo_traded = trade_price > 0 and int(ob.high_bidder[step]) == 1

            # Deduce action type from price
            action_type = "UNKNOWN"
            if ppo_price <= 0:
                action_type = "PASS"
            elif trade_price > 0 and ppo_traded:
                action_type = "ACCEPT/TRADE"
                trade_timing.append(step)
            elif spread > 0:
                if ppo_price == ppo_val:
                    action_type = "TRUTHFUL"
                elif ppo_price == high_bid + 1:
                    action_type = "JUMP_BEST"
                elif ppo_price > high_bid:
                    pct = (ppo_price - high_bid) / spread * 100
                    action_type = f"Improve ~{pct:.0f}%"
                else:
                    shade = (ppo_val - ppo_price) / ppo_val * 100 if ppo_val > 0 else 0
                    action_type = f"Shade ~{shade:.0f}%"
            else:
                if ppo_price == ppo_val:
                    action_type = "TRUTHFUL"
                else:
                    shade = (ppo_val - ppo_price) / ppo_val * 100 if ppo_val > 0 else 0
                    action_type = f"Shade ~{shade:.0f}%"

            # Count action types (simplified)
            if "PASS" in action_type:
                action_counts["PASS"] += 1
            elif "ACCEPT" in action_type or "TRADE" in action_type:
                action_counts["ACCEPT/TRADE"] += 1
            elif "TRUTHFUL" in action_type:
                action_counts["TRUTHFUL"] += 1
            elif "JUMP" in action_type:
                action_counts["JUMP_BEST"] += 1
            elif "Improve" in action_type:
                action_counts["Improve"] += 1
            elif "Shade" in action_type:
                action_counts["Shade"] += 1

            trace = {
                "step": step,
                "high_bid": high_bid,
                "low_ask": low_ask,
                "spread": spread,
                "ppo_val": ppo_val,
                "ppo_tokens": ppo_tokens_left,
                "ppo_price": ppo_price,
                "action_type": action_type,
                "traded": ppo_traded,
                "trade_price": trade_price if ppo_traded else 0,
            }
            period_traces.append(trace)

            if verbose and (ppo_price > 0 or ppo_traded):
                profit = ppo_val - trade_price if ppo_traded else 0
                print(
                    f"t={step:3d} | bid={high_bid:4d} ask={low_ask:4d} spread={spread:4d} | "
                    f"val={ppo_val:4d} tokens={ppo_tokens_left} | "
                    f"PPO={ppo_price:4d} [{action_type:15s}]"
                    + (f" -> TRADE@{trade_price} profit={profit}" if ppo_traded else "")
                )

        all_traces.append(period_traces)

        if verbose:
            print(
                f"\nPeriod {period} Summary: PPO profit={ppo.period_profit}, trades={ppo.num_trades}"
            )

    return all_traces, action_counts, trade_timing


def run_multi_seed_analysis(
    seeds, num_periods=5, verbose=False, opponent_type="zic", model_path=None
):
    """Run analysis across multiple seeds."""
    all_action_counts = defaultdict(int)
    all_trade_timing = []
    all_profits = []
    all_trades = []
    shade_values = []

    for seed in seeds:
        traces, action_counts, trade_timing = run_traced_market(
            seed=seed,
            num_periods=num_periods,
            verbose=verbose,
            opponent_type=opponent_type,
            model_path=model_path,
        )

        for k, v in action_counts.items():
            all_action_counts[k] += v
        all_trade_timing.extend(trade_timing)

        # Calculate profit and trades from traces
        for period_traces in traces:
            period_profit = sum(t.get("trade_price", 0) for t in period_traces if t.get("traded"))
            trades_in_period = sum(1 for t in period_traces if t.get("traded"))

            # Extract shade percentages
            for t in period_traces:
                if "Shade" in t.get("action_type", ""):
                    val = t.get("ppo_val", 0)
                    price = t.get("ppo_price", 0)
                    if val > 0 and price > 0:
                        shade_pct = (val - price) / val * 100
                        shade_values.append(shade_pct)

    return all_action_counts, all_trade_timing, shade_values


def print_analysis_results(action_counts, trade_timing, shade_values, label):
    """Print analysis results for a given opponent type."""
    print(f"\n{'='*70}")
    print(f"RESULTS: {label}")
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
        bins = [0, 5, 10, 20, 30, 40, 50]
        print("  Distribution:")
        for i in range(len(bins) - 1):
            count = sum(1 for s in shade_values if bins[i] <= s < bins[i + 1])
            print(
                f"    {bins[i]:2d}-{bins[i+1]:2d}%: {count:4d} ({count/len(shade_values)*100:5.1f}%)"
            )


def main():
    import argparse
    import json
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opponent",
        choices=["zic", "mixed", "both"],
        default="both",
        help="Opponent type: zic, mixed, or both for comparison",
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--periods", type=int, default=5, help="Periods per seed")
    parser.add_argument(
        "--model",
        type=str,
        default="checkpoints/ppo_v10_10M/ppo_double_auction_8000000_steps.zip",
        help="Path to PPO model checkpoint",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (optional)")
    args = parser.parse_args()

    print("=" * 70)
    print("PPO TRADING BEHAVIOR ANALYSIS")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 1000][: args.seeds]
    num_periods = args.periods

    print(
        f"\nConfig: {len(seeds)} seeds x {num_periods} periods = {len(seeds)*num_periods} periods total"
    )
    print(f"Seeds: {seeds}")
    print(f"Model: {args.model}")

    results = {}

    json_output = {
        "config": {
            "seeds": seeds,
            "periods": num_periods,
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
        }
    }

    if args.opponent in ["zic", "both"]:
        print("\n" + "=" * 70)
        print("ANALYZING: PPO vs ZIC OPPONENTS")
        print("=" * 70)
        action_counts, trade_timing, shade_values = run_multi_seed_analysis(
            seeds,
            num_periods=num_periods,
            verbose=False,
            opponent_type="zic",
            model_path=args.model,
        )
        results["zic"] = (action_counts, trade_timing, shade_values)
        print_analysis_results(action_counts, trade_timing, shade_values, "PPO vs ZIC")

        # Store for JSON output
        total = sum(action_counts.values())
        json_output["zic"] = {
            "action_distribution": {
                k: {"count": v, "pct": v / total * 100} for k, v in action_counts.items()
            },
            "trade_timing": {
                "total_trades": len(trade_timing),
                "mean_time": float(np.mean(trade_timing)) if trade_timing else 0,
                "std_time": float(np.std(trade_timing)) if trade_timing else 0,
                "early_pct": (
                    sum(1 for t in trade_timing if t < 30) / len(trade_timing) * 100
                    if trade_timing
                    else 0
                ),
                "mid_pct": (
                    sum(1 for t in trade_timing if 30 <= t < 70) / len(trade_timing) * 100
                    if trade_timing
                    else 0
                ),
                "late_pct": (
                    sum(1 for t in trade_timing if t >= 70) / len(trade_timing) * 100
                    if trade_timing
                    else 0
                ),
            },
            "shade_analysis": {
                "mean_shade": float(np.mean(shade_values)) if shade_values else 0,
                "std_shade": float(np.std(shade_values)) if shade_values else 0,
            },
        }

    if args.opponent in ["mixed", "both"]:
        print("\n" + "=" * 70)
        print("ANALYZING: PPO vs MIXED OPPONENTS (Skeleton, ZIP, Kaplan)")
        print("=" * 70)
        action_counts, trade_timing, shade_values = run_multi_seed_analysis(
            seeds,
            num_periods=num_periods,
            verbose=False,
            opponent_type="mixed",
            model_path=args.model,
        )
        results["mixed"] = (action_counts, trade_timing, shade_values)
        print_analysis_results(action_counts, trade_timing, shade_values, "PPO vs MIXED")

        # Store for JSON output
        total = sum(action_counts.values())
        json_output["mixed"] = {
            "action_distribution": {
                k: {"count": v, "pct": v / total * 100} for k, v in action_counts.items()
            },
            "trade_timing": {
                "total_trades": len(trade_timing),
                "mean_time": float(np.mean(trade_timing)) if trade_timing else 0,
                "std_time": float(np.std(trade_timing)) if trade_timing else 0,
                "early_pct": (
                    sum(1 for t in trade_timing if t < 30) / len(trade_timing) * 100
                    if trade_timing
                    else 0
                ),
                "mid_pct": (
                    sum(1 for t in trade_timing if 30 <= t < 70) / len(trade_timing) * 100
                    if trade_timing
                    else 0
                ),
                "late_pct": (
                    sum(1 for t in trade_timing if t >= 70) / len(trade_timing) * 100
                    if trade_timing
                    else 0
                ),
            },
            "shade_analysis": {
                "mean_shade": float(np.mean(shade_values)) if shade_values else 0,
                "std_shade": float(np.std(shade_values)) if shade_values else 0,
            },
        }

    # Print comparison if both
    if args.opponent == "both" and "zic" in results and "mixed" in results:
        print("\n" + "=" * 70)
        print("COMPARISON: ZIC vs MIXED OPPONENTS")
        print("=" * 70)

        zic_actions, zic_timing, zic_shade = results["zic"]
        mix_actions, mix_timing, mix_shade = results["mixed"]

        print("\n                     ZIC          MIXED        CHANGE")
        print("-" * 60)

        # Compare shade %
        zic_total = sum(zic_actions.values())
        mix_total = sum(mix_actions.values())
        zic_shade_pct = zic_actions.get("Shade", 0) / zic_total * 100
        mix_shade_pct = mix_actions.get("Shade", 0) / mix_total * 100
        print(
            f"Shade actions:      {zic_shade_pct:5.1f}%       {mix_shade_pct:5.1f}%       {mix_shade_pct - zic_shade_pct:+.1f}pp"
        )

        # Compare trade timing
        if zic_timing and mix_timing:
            zic_early = sum(1 for t in zic_timing if t < 30) / len(zic_timing) * 100
            mix_early = sum(1 for t in mix_timing if t < 30) / len(mix_timing) * 100
            print(
                f"Early trades:       {zic_early:5.1f}%       {mix_early:5.1f}%       {mix_early - zic_early:+.1f}pp"
            )

            zic_mean = np.mean(zic_timing)
            mix_mean = np.mean(mix_timing)
            print(
                f"Mean trade time:    {zic_mean:5.1f}        {mix_mean:5.1f}        {mix_mean - zic_mean:+.1f}"
            )

        # Compare shade depth
        if zic_shade and mix_shade:
            zic_mean_shade = np.mean(zic_shade)
            mix_mean_shade = np.mean(mix_shade)
            print(
                f"Mean shade %:       {zic_mean_shade:5.1f}%       {mix_mean_shade:5.1f}%       {mix_mean_shade - zic_mean_shade:+.1f}pp"
            )

    # Save to JSON if output path specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(json_output, f, indent=2)
        print(f"\n{'='*70}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
