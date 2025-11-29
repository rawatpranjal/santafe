"""
Phase 1 Task 1.4: Qualitative Manual Inspection Tests

This module provides tests that run realistic market scenarios and print
human-readable traces for manual inspection of market dynamics.

Run with: pytest tests/phase1/test_task_1_4_qualitative_inspection.py -v -s

What to look for:
1. Spread Convergence: Initial spread should be wide, then narrow
2. Price Discovery: Trade prices should cluster around competitive equilibrium
3. Efficient Ordering: High-surplus trades should execute before low-surplus
4. No Anomalies: No trades at absurd prices, no stuck markets

References:
- PLAN.md Phase 1 success criteria
- Gode & Sunder (1993) for ZIC efficiency benchmarks (~98%)
"""

import pytest
import numpy as np
from typing import List, Dict

from engine.orderbook import OrderBook
from engine.market import Market
from engine.visual_tracer import extract_market_timeline, TimestepRecord
from engine.efficiency import (
    calculate_max_surplus,
    calculate_actual_surplus,
    calculate_allocative_efficiency,
    extract_trades_from_orderbook,
    get_transaction_prices,
)
from traders.legacy.zic import ZIC
from traders.legacy.kaplan import Kaplan
from traders.legacy.skeleton import Skeleton
from traders.legacy.gradual import GradualBidder


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def calculate_theoretical_ce(
    buyer_vals: List[List[int]], seller_costs: List[List[int]]
) -> int:
    """Calculate approximate competitive equilibrium price."""
    # Flatten and sort
    all_bids = sorted([v for vals in buyer_vals for v in vals], reverse=True)
    all_asks = sorted([c for costs in seller_costs for c in costs])

    # Find crossing point
    for i in range(min(len(all_bids), len(all_asks))):
        if all_bids[i] <= all_asks[i]:
            # CE is between last profitable pair
            if i > 0:
                return (all_bids[i - 1] + all_asks[i - 1]) // 2
            return (all_bids[0] + all_asks[0]) // 2
    # All pairs profitable
    return (all_bids[-1] + all_asks[-1]) // 2


def print_market_header(
    num_buyers: int,
    num_sellers: int,
    num_timesteps: int,
    buyer_vals: List[List[int]],
    seller_costs: List[List[int]],
) -> None:
    """Print market setup information."""
    print("\n" + "=" * 70)
    print("QUALITATIVE MARKET INSPECTION")
    print("=" * 70)
    print(f"Market: {num_buyers} buyers x {num_sellers} sellers | {num_timesteps} timesteps | ZIC self-play")
    print()
    print(f"Buyer valuations:  {[v[0] for v in buyer_vals]}")
    print(f"Seller costs:      {[c[0] for c in seller_costs]}")
    ce = calculate_theoretical_ce(buyer_vals, seller_costs)
    print(f"Theoretical CE:    ~{ce} (where supply meets demand)")

    # Count max profitable trades
    all_bids = sorted([v for vals in buyer_vals for v in vals], reverse=True)
    all_asks = sorted([c for costs in seller_costs for c in costs])
    max_trades = sum(1 for b, a in zip(all_bids, all_asks) if b > a)
    print(f"Max possible trades: {max_trades} (buyers with val > lowest seller cost)")
    print()


def print_timestep_trace(timeline: List[TimestepRecord], max_rows: int = 30) -> None:
    """Print timestep-by-timestep market evolution."""
    print("-" * 70)
    print("TIMESTEP-BY-TIMESTEP EVOLUTION")
    print("-" * 70)
    print(f"{'t':>3} | {'Bid':>6} | {'Ask':>6} | {'Spread':>6} | Result")
    print("-" * 70)

    rows_printed = 0
    last_trade_t = 0

    for record in timeline:
        # Determine result string
        if record.trade_occurred:
            result = f"TRADE @ {record.trade_price}"
            last_trade_t = record.time
        elif record.high_bid == 0 and record.low_ask == 0:
            result = "[Book empty]"
        elif record.spread > 0:
            result = "No trade (spread > 0)"
        else:
            result = "No trade"

        # Format bid/ask display
        bid_str = str(record.high_bid) if record.high_bid > 0 else "-"
        ask_str = str(record.low_ask) if record.low_ask > 0 else "-"
        spread_str = str(record.spread) if record.spread > 0 else "-"

        # Print significant rows (trades, first few, or spread changes)
        is_significant = (
            record.trade_occurred
            or record.time <= 5
            or record.time == len(timeline)
            or rows_printed < max_rows
        )

        if is_significant:
            print(f"{record.time:>3} | {bid_str:>6} | {ask_str:>6} | {spread_str:>6} | {result}")
            rows_printed += 1

        if rows_printed == max_rows and record.time < len(timeline) - 5:
            print(f"... (skipping {len(timeline) - record.time - 5} timesteps) ...")
            rows_printed += 1

    print()


def print_market_summary(
    orderbook: OrderBook,
    num_times: int,
    buyer_vals: List[List[int]],
    seller_costs: List[List[int]],
) -> Dict[str, float]:
    """Print market summary statistics and return metrics."""
    print("-" * 70)
    print("MARKET SUMMARY")
    print("-" * 70)

    # Extract trades
    trades = extract_trades_from_orderbook(orderbook, num_times)
    total_trades = len(trades)

    # Calculate efficiency
    buyer_vals_dict = {i + 1: vals for i, vals in enumerate(buyer_vals)}
    seller_costs_dict = {i + 1: costs for i, costs in enumerate(seller_costs)}

    max_surplus = calculate_max_surplus(buyer_vals, seller_costs)
    actual_surplus = calculate_actual_surplus(trades, buyer_vals_dict, seller_costs_dict)
    efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

    # Get prices
    prices = get_transaction_prices(orderbook, num_times)
    avg_price = np.mean(prices) if prices else 0

    # Calculate spread evolution
    initial_spread = 0
    final_spread = 0
    for t in range(1, num_times + 1):
        hb = int(orderbook.high_bid[t])
        la = int(orderbook.low_ask[t])
        if hb > 0 and la > 0:
            spread = la - hb
            if initial_spread == 0:
                initial_spread = spread
            final_spread = spread

    ce = calculate_theoretical_ce(buyer_vals, seller_costs)

    print(f"Total trades:      {total_trades}")
    print(f"Efficiency:        {efficiency:.1f}%")
    print(f"Avg trade price:   {avg_price:.1f} (CE ~{ce})")
    print(f"Spread evolution:  {initial_spread} -> {final_spread}")
    print()

    return {
        "trades": total_trades,
        "efficiency": efficiency,
        "avg_price": avg_price,
        "ce": ce,
        "initial_spread": initial_spread,
        "final_spread": final_spread,
    }


def print_qualitative_checks(metrics: Dict[str, float]) -> None:
    """Print qualitative check results."""
    print("-" * 70)
    print("QUALITATIVE CHECKS")
    print("-" * 70)

    checks = []

    # Check 1: Spread narrowed
    if metrics["initial_spread"] > 0 and metrics["final_spread"] < metrics["initial_spread"]:
        checks.append(("Spread narrowed over time", True))
    elif metrics["initial_spread"] == 0:
        checks.append(("Spread narrowed over time", None))  # N/A
    else:
        checks.append(("Spread narrowed over time", False))

    # Check 2: Prices near CE
    if metrics["avg_price"] > 0:
        price_deviation = abs(metrics["avg_price"] - metrics["ce"]) / metrics["ce"] * 100
        checks.append((f"Prices within 20% of CE ({price_deviation:.1f}% off)", price_deviation < 20))
    else:
        checks.append(("Prices within 20% of CE", None))

    # Check 3: At least one trade
    checks.append(("At least one trade executed", metrics["trades"] > 0))

    # Check 4: Efficiency > 80%
    checks.append((f"Efficiency > 80% (got {metrics['efficiency']:.1f}%)", metrics["efficiency"] > 80))

    for check, passed in checks:
        if passed is None:
            print(f"  ? {check} (N/A)")
        elif passed:
            print(f"  [PASS] {check}")
        else:
            print(f"  [FAIL] {check}")

    print()
    print("=" * 70)


# =============================================================================
# TEST CLASS
# =============================================================================


class TestQualitativeMarketInspection:
    """
    Qualitative inspection tests that print market evolution for human review.

    Run with: pytest tests/phase1/test_task_1_4_qualitative_inspection.py -v -s

    These tests have minimal assertions - the primary purpose is visual inspection.
    """

    def test_zic_market_evolution_trace(self) -> None:
        """
        Run a 4x4 ZIC market and print timestep-by-timestep evolution.

        This is the main qualitative inspection test. It runs a realistic
        market scenario and prints:
        1. Market setup (valuations, costs, theoretical CE)
        2. Timestep-by-timestep evolution (bid, ask, spread, trades)
        3. Summary statistics (efficiency, avg price, trade count)
        4. Qualitative checks (spread convergence, price discovery)
        """
        # Setup: 4 buyers x 4 sellers with known valuations
        num_buyers = 4
        num_sellers = 4
        num_tokens = 1
        num_timesteps = 50
        price_min = 1
        price_max = 200

        # Valuations designed for clear equilibrium
        # Buyers: [180, 160, 140, 120] - sorted descending
        # Sellers: [80, 100, 120, 140] - sorted ascending
        # CE around 130, max 3 profitable trades (180>80, 160>100, 140>120)
        buyer_vals = [[180], [160], [140], [120]]
        seller_costs = [[80], [100], [120], [140]]

        # Create agents
        buyers: List[ZIC] = []
        sellers: List[ZIC] = []

        for i in range(num_buyers):
            buyers.append(
                ZIC(
                    player_id=i + 1,
                    is_buyer=True,
                    num_tokens=num_tokens,
                    valuations=buyer_vals[i],
                    price_min=price_min,
                    price_max=price_max,
                    seed=42 + i,  # Reproducible
                )
            )

        for i in range(num_sellers):
            sellers.append(
                ZIC(
                    player_id=i + 1,
                    is_buyer=False,
                    num_tokens=num_tokens,
                    valuations=seller_costs[i],
                    price_min=price_min,
                    price_max=price_max,
                    seed=100 + i,  # Reproducible
                )
            )

        # Create market
        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            num_times=num_timesteps,
            price_min=price_min,
            price_max=price_max,
            buyers=buyers,
            sellers=sellers,
        )

        # Run market
        for _ in range(num_timesteps):
            success = market.run_time_step()
            if not success:
                break

        # Print traces
        print_market_header(num_buyers, num_sellers, num_timesteps, buyer_vals, seller_costs)

        # Extract timeline
        buyer_types = {i + 1: "ZIC" for i in range(num_buyers)}
        seller_types = {i + 1: "ZIC" for i in range(num_sellers)}
        timeline = extract_market_timeline(market.orderbook, buyer_types, seller_types)

        print_timestep_trace(timeline)
        metrics = print_market_summary(market.orderbook, num_timesteps, buyer_vals, seller_costs)
        print_qualitative_checks(metrics)

        # Minimal assertions - test passes if market runs
        assert metrics["trades"] >= 0, "Market should run without errors"
        assert metrics["efficiency"] >= 0, "Efficiency should be non-negative"

    def test_spread_convergence_visual(self) -> None:
        """
        Show spread narrowing over time with ASCII visualization.
        """
        # Setup: 4x4 market
        num_buyers = 4
        num_sellers = 4
        num_tokens = 1
        num_timesteps = 40
        price_min = 1
        price_max = 200

        buyer_vals = [[180], [160], [140], [120]]
        seller_costs = [[80], [100], [120], [140]]

        buyers = [
            ZIC(i + 1, True, num_tokens, buyer_vals[i], price_min, price_max, seed=42 + i)
            for i in range(num_buyers)
        ]
        sellers = [
            ZIC(i + 1, False, num_tokens, seller_costs[i], price_min, price_max, seed=100 + i)
            for i in range(num_sellers)
        ]

        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            num_times=num_timesteps,
            price_min=price_min,
            price_max=price_max,
            buyers=buyers,
            sellers=sellers,
        )

        for _ in range(num_timesteps):
            market.run_time_step()

        # Collect spreads
        spreads = []
        for t in range(1, num_timesteps + 1):
            hb = int(market.orderbook.high_bid[t])
            la = int(market.orderbook.low_ask[t])
            if hb > 0 and la > 0:
                spreads.append(la - hb)
            else:
                spreads.append(None)

        # Print ASCII spread chart
        print("\n" + "=" * 50)
        print("SPREAD EVOLUTION (ASCII)")
        print("=" * 50)

        max_spread = max(s for s in spreads if s is not None) if any(spreads) else 100
        chart_width = 40

        for t, spread in enumerate(spreads[:20], 1):  # First 20 timesteps
            if spread is not None:
                bar_len = int((spread / max_spread) * chart_width)
                bar = "#" * bar_len
                print(f"t={t:>2} | {bar:<{chart_width}} | {spread}")
            else:
                print(f"t={t:>2} | {'[no spread]':<{chart_width}} | -")

        print()

        # Check spread narrowed
        valid_spreads = [s for s in spreads if s is not None]
        if len(valid_spreads) >= 2:
            initial = valid_spreads[0]
            final = valid_spreads[-1]
            print(f"Initial spread: {initial}")
            print(f"Final spread: {final}")
            # Don't assert - just report
        print()

    def test_trade_price_distribution(self) -> None:
        """
        Show where trades executed relative to equilibrium.
        """
        # Run a larger market with multiple tokens for more trades
        num_buyers = 4
        num_sellers = 4
        num_tokens = 2
        num_timesteps = 80
        price_min = 1
        price_max = 200

        # 2 tokens each = up to 8 trades possible
        buyer_vals = [[180, 150], [170, 145], [160, 135], [140, 125]]
        seller_costs = [[70, 90], [80, 100], [95, 115], [110, 130]]

        buyers = [
            ZIC(i + 1, True, num_tokens, buyer_vals[i], price_min, price_max, seed=42 + i)
            for i in range(num_buyers)
        ]
        sellers = [
            ZIC(i + 1, False, num_tokens, seller_costs[i], price_min, price_max, seed=100 + i)
            for i in range(num_sellers)
        ]

        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            num_times=num_timesteps,
            price_min=price_min,
            price_max=price_max,
            buyers=buyers,
            sellers=sellers,
        )

        for _ in range(num_timesteps):
            market.run_time_step()

        # Get trade prices
        prices = get_transaction_prices(market.orderbook, num_timesteps)
        ce = calculate_theoretical_ce(buyer_vals, seller_costs)

        print("\n" + "=" * 50)
        print("TRADE PRICE DISTRIBUTION")
        print("=" * 50)
        print(f"Theoretical CE: ~{ce}")
        print(f"Total trades: {len(prices)}")
        print()

        if prices:
            # Price histogram (simple text)
            print("Trade prices:")
            for i, p in enumerate(prices, 1):
                deviation = p - ce
                direction = "+" if deviation > 0 else ""
                print(f"  Trade {i}: {p} ({direction}{deviation} from CE)")

            print()
            print(f"Mean price:  {np.mean(prices):.1f}")
            print(f"Std dev:     {np.std(prices):.1f}")
            print(f"Min:         {min(prices)}")
            print(f"Max:         {max(prices)}")
        else:
            print("No trades executed!")

        print()

    def test_multi_period_consistency(self) -> None:
        """
        Run 3 periods and show consistency across periods.
        """
        print("\n" + "=" * 60)
        print("MULTI-PERIOD CONSISTENCY CHECK")
        print("=" * 60)

        num_buyers = 4
        num_sellers = 4
        num_tokens = 1
        num_timesteps = 40
        price_min = 1
        price_max = 200
        num_periods = 3

        buyer_vals = [[180], [160], [140], [120]]
        seller_costs = [[80], [100], [120], [140]]

        results = []

        for period in range(num_periods):
            # Fresh agents each period (different seeds)
            buyers = [
                ZIC(i + 1, True, num_tokens, buyer_vals[i], price_min, price_max, seed=period * 100 + i)
                for i in range(num_buyers)
            ]
            sellers = [
                ZIC(i + 1, False, num_tokens, seller_costs[i], price_min, price_max, seed=period * 100 + 50 + i)
                for i in range(num_sellers)
            ]

            market = Market(
                num_buyers=num_buyers,
                num_sellers=num_sellers,
                num_times=num_timesteps,
                price_min=price_min,
                price_max=price_max,
                buyers=buyers,
                sellers=sellers,
            )

            for _ in range(num_timesteps):
                market.run_time_step()

            # Calculate metrics
            trades = extract_trades_from_orderbook(market.orderbook, num_timesteps)
            buyer_vals_dict = {i + 1: buyer_vals[i] for i in range(num_buyers)}
            seller_costs_dict = {i + 1: seller_costs[i] for i in range(num_sellers)}

            max_surplus = calculate_max_surplus(buyer_vals, seller_costs)
            actual_surplus = calculate_actual_surplus(trades, buyer_vals_dict, seller_costs_dict)
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)

            prices = get_transaction_prices(market.orderbook, num_timesteps)
            avg_price = np.mean(prices) if prices else 0

            results.append({
                "period": period + 1,
                "trades": len(trades),
                "efficiency": efficiency,
                "avg_price": avg_price,
            })

        # Print results
        print()
        print(f"{'Period':>8} | {'Trades':>8} | {'Efficiency':>12} | {'Avg Price':>10}")
        print("-" * 50)
        for r in results:
            print(f"{r['period']:>8} | {r['trades']:>8} | {r['efficiency']:>11.1f}% | {r['avg_price']:>10.1f}")

        print("-" * 50)

        # Summary
        efficiencies = [r["efficiency"] for r in results]
        print(f"{'Mean':>8} |          | {np.mean(efficiencies):>11.1f}% |")
        print(f"{'Std':>8} |          | {np.std(efficiencies):>11.1f}% |")
        print()

        # Check consistency
        if np.std(efficiencies) < 15:
            print("[PASS] Efficiency consistent across periods (std < 15%)")
        else:
            print("[WARN] High variance in efficiency across periods")
        print()

    def test_kaplan_vs_skeleton_sniper_behavior(self) -> None:
        """
        Run 1 Kaplan buyer + 7 Skeleton agents to observe sniper behavior.

        This test demonstrates Kaplan's "sniper" strategy:
        1. Waits while Skeleton agents narrow the spread
        2. Jumps in when spread < 10% or time runs out
        3. Executes trades in final timesteps
        4. Extracts parasitic profit from passive traders
        """
        print("\n" + "=" * 70)
        print("KAPLAN SNIPER INSPECTION")
        print("=" * 70)
        print("Market: 1 Kaplan buyer + 3 Skeleton buyers + 4 Skeleton sellers")
        print()

        num_buyers = 4
        num_sellers = 4
        num_tokens = 1
        num_timesteps = 60
        price_min = 1
        price_max = 200

        # Valuations: Kaplan gets the best token (180)
        buyer_vals = [[180], [160], [140], [120]]
        seller_costs = [[80], [100], [120], [140]]

        print(f"Buyer valuations:  {[v[0] for v in buyer_vals]} (Kaplan=180)")
        print(f"Seller costs:      {[c[0] for c in seller_costs]}")
        ce = calculate_theoretical_ce(buyer_vals, seller_costs)
        print(f"Theoretical CE:    ~{ce}")
        print()

        # Create agents: 1 Kaplan buyer (id=1), 3 Skeleton buyers (ids 2-4)
        buyers: List = []

        # Buyer 1: Kaplan (the sniper)
        buyers.append(
            Kaplan(
                player_id=1,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=buyer_vals[0],
                price_min=price_min,
                price_max=price_max,
                num_times=num_timesteps,
            )
        )

        # Buyers 2-4: Skeleton (passive)
        for i in range(1, num_buyers):
            buyers.append(
                Skeleton(
                    player_id=i + 1,
                    is_buyer=True,
                    num_tokens=num_tokens,
                    valuations=buyer_vals[i],
                    price_min=price_min,
                    price_max=price_max,
                    num_times=num_timesteps,
                    seed=42 + i,
                )
            )

        # All sellers are Skeleton
        sellers: List = []
        for i in range(num_sellers):
            sellers.append(
                Skeleton(
                    player_id=i + 1,
                    is_buyer=False,
                    num_tokens=num_tokens,
                    valuations=seller_costs[i],
                    price_min=price_min,
                    price_max=price_max,
                    num_times=num_timesteps,
                    seed=100 + i,
                )
            )

        # Create market
        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            num_times=num_timesteps,
            price_min=price_min,
            price_max=price_max,
            buyers=buyers,
            sellers=sellers,
        )

        # Run market
        for _ in range(num_timesteps):
            success = market.run_time_step()
            if not success:
                break

        # Extract timeline
        buyer_types = {1: "Kaplan", 2: "Skel", 3: "Skel", 4: "Skel"}
        seller_types = {1: "Skel", 2: "Skel", 3: "Skel", 4: "Skel"}
        timeline = extract_market_timeline(market.orderbook, buyer_types, seller_types)

        # Print timestep trace with Kaplan highlighting
        print("-" * 70)
        print("TIMESTEP-BY-TIMESTEP EVOLUTION")
        print("-" * 70)
        print(f"{'t':>3} | {'Bidder':>8} | {'Bid':>5} | {'Asker':>8} | {'Ask':>5} | {'Spread':>6} | Result")
        print("-" * 70)

        kaplan_jump_in_time = None
        kaplan_trade_time = None
        kaplan_trade_price = None

        for record in timeline:
            # Find who has the high bid
            high_bidder_type = "---"
            high_bidder_id = int(market.orderbook.high_bidder[record.time])
            if high_bidder_id > 0:
                high_bidder_type = buyer_types.get(high_bidder_id, "?")

            # Find who has the low ask
            low_asker_type = "---"
            low_asker_id = int(market.orderbook.low_asker[record.time])
            if low_asker_id > 0:
                low_asker_type = seller_types.get(low_asker_id, "?")

            # Determine result
            if record.trade_occurred:
                result = f"TRADE @ {record.trade_price}"
                if high_bidder_id == 1:  # Kaplan traded
                    result += " [KAPLAN SNIPES!]"
                    kaplan_trade_time = record.time
                    kaplan_trade_price = record.trade_price
            elif record.spread > 0:
                result = f"No trade"
            else:
                result = "No trade"

            # Check if Kaplan became high bidder (jump in)
            if high_bidder_id == 1 and kaplan_jump_in_time is None:
                kaplan_jump_in_time = record.time
                result += " [KAPLAN JUMPS IN!]"

            # Format values
            bid_str = str(record.high_bid) if record.high_bid > 0 else "-"
            ask_str = str(record.low_ask) if record.low_ask > 0 else "-"
            spread_str = str(record.spread) if record.spread > 0 else "-"

            # Print key timesteps
            is_key = (
                record.time <= 5
                or record.trade_occurred
                or high_bidder_id == 1  # Kaplan is bidding
                or record.time >= num_timesteps - 5
                or (record.time % 10 == 0)
            )

            if is_key:
                print(
                    f"{record.time:>3} | {high_bidder_type:>8} | {bid_str:>5} | "
                    f"{low_asker_type:>8} | {ask_str:>5} | {spread_str:>6} | {result}"
                )

        print()

        # Kaplan behavior analysis
        print("-" * 70)
        print("KAPLAN BEHAVIOR ANALYSIS")
        print("-" * 70)

        if kaplan_jump_in_time:
            print(f"Kaplan jumped in at t={kaplan_jump_in_time}")
        else:
            print("Kaplan never became high bidder (stayed passive)")

        if kaplan_trade_time:
            remaining = num_timesteps - kaplan_trade_time
            print(f"Kaplan traded at t={kaplan_trade_time} ({remaining} steps from end)")
            kaplan_profit = buyer_vals[0][0] - kaplan_trade_price
            print(f"Kaplan profit: {kaplan_profit} (valuation {buyer_vals[0][0]} - price {kaplan_trade_price})")
        else:
            print("Kaplan did not trade")

        # Calculate all profits
        print()
        print("-" * 70)
        print("PROFIT DISTRIBUTION")
        print("-" * 70)

        print(f"{'Agent':>12} | {'Type':>8} | {'Trades':>6} | {'Profit':>8}")
        print("-" * 45)

        skeleton_profits = []
        for i, buyer in enumerate(buyers):
            agent_type = "Kaplan" if i == 0 else "Skeleton"
            print(f"Buyer {buyer.player_id:>5} | {agent_type:>8} | {buyer.num_trades:>6} | {buyer.period_profit:>8}")
            if i > 0:
                skeleton_profits.append(buyer.period_profit)

        for i, seller in enumerate(sellers):
            print(f"Seller {seller.player_id:>4} | {'Skeleton':>8} | {seller.num_trades:>6} | {seller.period_profit:>8}")
            skeleton_profits.append(seller.period_profit)

        print("-" * 45)

        kaplan_profit = buyers[0].period_profit
        avg_skeleton = np.mean(skeleton_profits) if skeleton_profits else 0

        print(f"Kaplan profit:       {kaplan_profit}")
        print(f"Avg Skeleton profit: {avg_skeleton:.1f}")

        if kaplan_profit > avg_skeleton:
            print("[PASS] Kaplan extracted more profit than average Skeleton (parasitic)")
        else:
            print("[INFO] Kaplan did not outperform Skeletons")

        print()
        print("=" * 70)

    def test_kaplan_vs_zic_sniper_success(self) -> None:
        """
        Run 1 Kaplan buyer as SOLE buyer against 4 ZIC sellers.

        This configuration demonstrates Kaplan's sniper strategy because:
        1. Kaplan is the only buyer, so always the high bidder
        2. ZIC sellers compete by lowering asks
        3. Kaplan waits for asks to drop, then strikes
        4. Time pressure in final timesteps forces trades

        Expected: Kaplan extracts good profit by waiting for low asks.
        """
        print("\n" + "=" * 70)
        print("KAPLAN SNIPER SUCCESS SCENARIO")
        print("=" * 70)
        print("Market: 1 Kaplan buyer (solo) vs 4 ZIC sellers")
        print("Timesteps: 100 (Kaplan observes, waits, strikes)")
        print()

        num_buyers = 1  # JUST KAPLAN
        num_sellers = 4
        num_tokens = 3  # Multiple opportunities
        num_timesteps = 100
        price_min = 1
        price_max = 200

        # Kaplan has high valuations - can afford to buy at good prices
        buyer_vals = [[180, 160, 140]]
        # Sellers have low costs - will compete downward
        seller_costs = [[40, 50, 60], [50, 60, 70], [60, 70, 80], [70, 80, 90]]

        print(f"Kaplan valuations: {buyer_vals[0]}")
        print(f"Seller costs:      {[c[0] for c in seller_costs]}")
        ce = calculate_theoretical_ce(buyer_vals, seller_costs)
        print(f"Theoretical CE:    ~{ce}")
        print()

        # Create agents: Just Kaplan as buyer
        buyers: List = []
        buyers.append(
            Kaplan(
                player_id=1,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=buyer_vals[0],
                price_min=price_min,
                price_max=price_max,
                num_times=num_timesteps,
            )
        )

        # All sellers are ZIC - they'll compete with each other
        sellers: List = []
        for i in range(num_sellers):
            sellers.append(
                ZIC(
                    player_id=i + 1,
                    is_buyer=False,
                    num_tokens=num_tokens,
                    valuations=seller_costs[i],
                    price_min=price_min,
                    price_max=price_max,
                    seed=100 + i,
                )
            )

        # Create market
        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            num_times=num_timesteps,
            price_min=price_min,
            price_max=price_max,
            buyers=buyers,
            sellers=sellers,
        )

        # Run market
        for _ in range(num_timesteps):
            success = market.run_time_step()
            if not success:
                break

        # Extract timeline
        buyer_types = {1: "Kaplan"}
        seller_types = {1: "ZIC", 2: "ZIC", 3: "ZIC", 4: "ZIC"}
        timeline = extract_market_timeline(market.orderbook, buyer_types, seller_types)

        # Print condensed timestep trace focusing on key moments
        print("-" * 70)
        print("KEY MARKET MOMENTS")
        print("-" * 70)
        print(f"{'t':>3} | {'Bidder':>8} | {'Bid':>5} | {'Asker':>8} | {'Ask':>5} | {'Spread':>6} | Result")
        print("-" * 70)

        kaplan_jump_in_time = None
        kaplan_trade_times = []
        kaplan_trade_prices = []
        last_printed_spread = None

        for record in timeline:
            high_bidder_id = int(market.orderbook.high_bidder[record.time])
            high_bidder_type = buyer_types.get(high_bidder_id, "---") if high_bidder_id > 0 else "---"

            low_asker_id = int(market.orderbook.low_asker[record.time])
            low_asker_type = seller_types.get(low_asker_id, "---") if low_asker_id > 0 else "---"

            # Build result string
            result = ""
            if record.trade_occurred:
                result = f"TRADE @ {record.trade_price}"
                if high_bidder_id == 1:
                    result += " [KAPLAN WINS!]"
                    kaplan_trade_times.append(record.time)
                    kaplan_trade_prices.append(record.trade_price)

            # Track Kaplan jump-in
            if high_bidder_id == 1 and kaplan_jump_in_time is None:
                kaplan_jump_in_time = record.time
                result += " [KAPLAN JUMPS IN!]"

            # Determine what to print
            is_key_moment = (
                record.time <= 3  # Start
                or record.trade_occurred
                or high_bidder_id == 1  # Kaplan active
                or record.time >= num_timesteps - 3  # End
                or (record.spread > 0 and record.spread != last_printed_spread and record.time % 20 == 0)  # Spread changes
            )

            if is_key_moment:
                bid_str = str(record.high_bid) if record.high_bid > 0 else "-"
                ask_str = str(record.low_ask) if record.low_ask > 0 else "-"
                spread_str = str(record.spread) if record.spread > 0 else "-"
                print(
                    f"{record.time:>3} | {high_bidder_type:>8} | {bid_str:>5} | "
                    f"{low_asker_type:>8} | {ask_str:>5} | {spread_str:>6} | {result}"
                )
                last_printed_spread = record.spread

        print()

        # Kaplan behavior analysis
        print("-" * 70)
        print("KAPLAN BEHAVIOR ANALYSIS")
        print("-" * 70)

        if kaplan_jump_in_time:
            print(f"Kaplan jumped in at t={kaplan_jump_in_time} (out of {num_timesteps})")
            pct = kaplan_jump_in_time / num_timesteps * 100
            print(f"  -> Waited for {pct:.0f}% of the period before acting")
        else:
            print("Kaplan never became high bidder")

        if kaplan_trade_times:
            print(f"Kaplan traded {len(kaplan_trade_times)} time(s):")
            for t, p in zip(kaplan_trade_times, kaplan_trade_prices):
                remaining = num_timesteps - t
                print(f"  -> t={t} (price={p}, {remaining} steps from end)")
        else:
            print("Kaplan did not trade")

        # Profit distribution
        print()
        print("-" * 70)
        print("PROFIT DISTRIBUTION")
        print("-" * 70)

        print(f"{'Agent':>12} | {'Type':>8} | {'Trades':>6} | {'Profit':>8}")
        print("-" * 45)

        zic_seller_profits = []
        kaplan_profit = 0

        # Kaplan is the only buyer
        kaplan_profit = buyers[0].period_profit
        print(f"Buyer     1 | {'Kaplan':>8} | {buyers[0].num_trades:>6} | {kaplan_profit:>8}")

        for i, seller in enumerate(sellers):
            profit = seller.period_profit
            print(f"Seller {seller.player_id:>4} | {'ZIC':>8} | {seller.num_trades:>6} | {profit:>8}")
            zic_seller_profits.append(profit)

        print("-" * 45)

        total_seller_profit = sum(zic_seller_profits)
        avg_seller = np.mean(zic_seller_profits) if zic_seller_profits else 0

        print(f"Kaplan profit:       {kaplan_profit}")
        print(f"Total seller profit: {total_seller_profit}")
        print(f"Avg seller profit:   {avg_seller:.1f}")

        print()
        if kaplan_profit > 0:
            # Calculate Kaplan's share of total surplus
            total_surplus = kaplan_profit + total_seller_profit
            kaplan_share = (kaplan_profit / total_surplus * 100) if total_surplus > 0 else 0
            print(f"[PASS] Kaplan traded successfully!")
            print(f"       Kaplan captured {kaplan_share:.0f}% of total surplus")
            if kaplan_profit > avg_seller:
                print(f"       Kaplan profit > avg seller profit (sniper advantage)")
        else:
            print("[INFO] Kaplan did not trade")

        print()
        print("=" * 70)

    def test_kaplan_deal_stealing_mechanic(self) -> None:
        """
        Demonstrate Kaplan's REAL "deal stealing" mechanic.

        This test shows the spread < 10% jump-in behavior from the JEDC paper:
        1. GradualBidder buyers narrow the spread through iterative bidding
        2. GradualBidder buyers NEVER accept trades (return False in buy_sell)
        3. When spread < 10%, Kaplan bids = CurrentAsk
        4. Kaplan becomes high bidder at a price the seller accepts
        5. Trade executes immediately - Kaplan "steals" the deal

        This is different from the TIME PRESSURE mechanic (final 2 timesteps).
        """
        print("\n" + "=" * 70)
        print("KAPLAN DEAL STEALING MECHANIC")
        print("=" * 70)
        print("Market: 1 Kaplan buyer + 3 GradualBidder buyers + 4 Skeleton sellers")
        print("GradualBidders narrow spread (bids), Skeletons narrow spread (asks)")
        print("Kaplan waits for spread < 10%, then bids = ask to 'steal' the deal")
        print()

        num_buyers = 4
        num_sellers = 4
        num_tokens = 3  # Multiple tokens = spread_val > 0 for gradual convergence
        num_timesteps = 100
        price_min = 1
        price_max = 200

        # Valuations designed for gradual convergence
        # Multiple tokens per agent with declining values (buyers) / increasing costs (sellers)
        # This creates spread_val > 0 which makes GradualBidder converge slowly
        buyer_vals = [[180, 160, 140], [170, 150, 130], [160, 140, 120], [150, 130, 110]]
        seller_costs = [[60, 80, 100], [70, 90, 110], [80, 100, 120], [90, 110, 130]]

        print(f"Buyer valuations:  {buyer_vals} (Kaplan first)")
        print(f"Seller costs:      {seller_costs}")
        ce = calculate_theoretical_ce(buyer_vals, seller_costs)
        print(f"Theoretical CE:    ~{ce}")
        print()

        # Create agents: 1 Kaplan + 3 GradualBidder buyers
        buyers: List = []

        # Buyer 1: Kaplan (the deal stealer)
        buyers.append(
            Kaplan(
                player_id=1,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=buyer_vals[0],
                price_min=price_min,
                price_max=price_max,
                num_times=num_timesteps,
            )
        )

        # Buyers 2-4: GradualBidder (narrow spread but never trade)
        for i in range(1, num_buyers):
            buyers.append(
                GradualBidder(
                    player_id=i + 1,
                    is_buyer=True,
                    num_tokens=num_tokens,
                    valuations=buyer_vals[i],
                    price_min=price_min,
                    price_max=price_max,
                    num_times=num_timesteps,
                    seed=42 + i,
                )
            )

        # Sellers are Skeleton - they will accept trades when profitable
        # This allows Kaplan to "steal" by bidding = ask (trade executes)
        sellers: List = []
        for i in range(num_sellers):
            sellers.append(
                Skeleton(
                    player_id=i + 1,
                    is_buyer=False,
                    num_tokens=num_tokens,
                    valuations=seller_costs[i],
                    price_min=price_min,
                    price_max=price_max,
                    num_times=num_timesteps,
                    seed=100 + i,
                )
            )

        # Create market
        market = Market(
            num_buyers=num_buyers,
            num_sellers=num_sellers,
            num_times=num_timesteps,
            price_min=price_min,
            price_max=price_max,
            buyers=buyers,
            sellers=sellers,
        )

        # Track spread evolution and Kaplan behavior
        spread_history = []
        kaplan_bid_history = []
        kaplan_became_bidder_at = None
        kaplan_steal_time = None
        kaplan_steal_price = None

        # Run market timestep by timestep
        kaplan_bids = []  # Track what Kaplan actually bids
        for t in range(1, num_timesteps + 1):
            success = market.run_time_step()

            # Track spread
            hb = int(market.orderbook.high_bid[t])
            la = int(market.orderbook.low_ask[t])
            spread = (la - hb) if (hb > 0 and la > 0) else None
            spread_history.append(spread)

            # Track Kaplan's actual bid at this timestep
            kaplan_bid_at_t = int(market.orderbook.bids[1, t])  # Player 1 is Kaplan
            kaplan_bids.append(kaplan_bid_at_t)

            # Track Kaplan's bid and whether it became high bidder
            kaplan_bid = 0
            for bid_time in range(1, t + 1):
                if int(market.orderbook.high_bidder[bid_time]) == 1:  # Kaplan is player 1
                    kaplan_bid = int(market.orderbook.high_bid[bid_time])
            kaplan_bid_history.append(kaplan_bid)

            # Check if Kaplan became high bidder
            high_bidder_id = int(market.orderbook.high_bidder[t])
            if high_bidder_id == 1 and kaplan_became_bidder_at is None:
                kaplan_became_bidder_at = t

            # Check if trade occurred and Kaplan was involved
            trade_price_at_t = int(market.orderbook.trade_price[t])
            if trade_price_at_t > 0:
                # Trade happened - check if Kaplan was the buyer
                # In AURORA, trade happens when high_bid >= low_ask and someone accepts
                if high_bidder_id == 1:
                    kaplan_steal_time = t
                    kaplan_steal_price = trade_price_at_t
                    break  # Stop after first steal

            if not success:
                break

        # Print spread evolution
        print("-" * 80)
        print("SPREAD EVOLUTION (tracking spread narrowing)")
        print("-" * 80)
        print(f"{'t':>3} | {'Bid':>5} | {'Ask':>5} | {'Spread':>6} | {'Bidder':>8} | {'Kap':>5} | Notes")
        print("-" * 80)

        buyer_types = {1: "Kaplan", 2: "Grad", 3: "Grad", 4: "Grad"}

        for t in range(1, min(num_timesteps + 1, len(spread_history) + 1)):
            hb = int(market.orderbook.high_bid[t])
            la = int(market.orderbook.low_ask[t])
            spread = spread_history[t - 1]
            high_bidder_id = int(market.orderbook.high_bidder[t])
            bidder_type = buyer_types.get(high_bidder_id, "---") if high_bidder_id > 0 else "---"

            # Build notes
            notes = ""
            trade_price_check = int(market.orderbook.trade_price[t])
            if trade_price_check > 0:
                if high_bidder_id == 1:
                    notes = f"TRADE @ {trade_price_check} [KAPLAN STEALS!]"
                else:
                    notes = f"TRADE @ {trade_price_check}"
            elif t == kaplan_became_bidder_at:
                notes = "[Kaplan jumps in!]"
            elif spread is not None and spread > 0:
                spread_pct = spread / la * 100 if la > 0 else 0
                if spread_pct < 10:
                    notes = f"(spread {spread_pct:.1f}% < 10%)"

            # Print key timesteps
            is_key = (
                t <= 5
                or t == kaplan_became_bidder_at
                or t == kaplan_steal_time
                or trade_price_check > 0
                or t >= num_timesteps - 3
                or t % 10 == 0
            )

            if is_key:
                bid_str = str(hb) if hb > 0 else "-"
                ask_str = str(la) if la > 0 else "-"
                spread_str = str(spread) if spread is not None else "-"
                kaplan_bid_str = str(kaplan_bids[t - 1]) if t <= len(kaplan_bids) else "-"
                print(f"{t:>3} | {bid_str:>5} | {ask_str:>5} | {spread_str:>6} | {bidder_type:>8} | K={kaplan_bid_str:>3} | {notes}")

        print()

        # Analysis
        print("-" * 70)
        print("KAPLAN BEHAVIOR ANALYSIS")
        print("-" * 70)

        if kaplan_became_bidder_at:
            # Calculate spread at that moment
            t = kaplan_became_bidder_at
            hb = int(market.orderbook.high_bid[t])
            la = int(market.orderbook.low_ask[t])
            spread_at_jump = la - hb if (hb > 0 and la > 0) else None
            spread_pct = (spread_at_jump / la * 100) if (spread_at_jump and la > 0) else 0
            print(f"Kaplan jumped in at t={kaplan_became_bidder_at}")
            if spread_at_jump:
                print(f"  Spread at jump: {spread_at_jump} ({spread_pct:.1f}%)")
                if spread_pct < 10:
                    print(f"  [PASS] Kaplan triggered <10% spread jump-in logic!")
                else:
                    print(f"  [INFO] Kaplan jumped in via different trigger (time pressure?)")
        else:
            print("Kaplan never became high bidder")

        if kaplan_steal_time:
            remaining = num_timesteps - kaplan_steal_time
            kaplan_profit = buyer_vals[0][0] - kaplan_steal_price
            print(f"Kaplan STOLE deal at t={kaplan_steal_time} ({remaining} steps from end)")
            print(f"  Trade price: {kaplan_steal_price}")
            print(f"  Kaplan profit: {kaplan_profit} (valuation {buyer_vals[0][0]} - price {kaplan_steal_price})")

            # Check if this was the "real" deal steal (spread < 10%, not time pressure)
            if remaining > 2:
                print(f"  [PASS] This was a REAL deal steal (not time pressure)!")
            else:
                print(f"  [INFO] Trade happened in final 2 steps (could be time pressure)")
        else:
            print("Kaplan did not trade")

        # Profit summary
        print()
        print("-" * 70)
        print("PROFIT DISTRIBUTION")
        print("-" * 70)

        print(f"{'Agent':>12} | {'Type':>10} | {'Trades':>6} | {'Profit':>8}")
        print("-" * 50)

        for i, buyer in enumerate(buyers):
            agent_type = "Kaplan" if i == 0 else "Gradual"
            print(f"Buyer {buyer.player_id:>5} | {agent_type:>10} | {buyer.num_trades:>6} | {buyer.period_profit:>8}")

        for i, seller in enumerate(sellers):
            print(f"Seller {seller.player_id:>4} | {'Skeleton':>10} | {seller.num_trades:>6} | {seller.period_profit:>8}")

        print()
        print("=" * 70)
