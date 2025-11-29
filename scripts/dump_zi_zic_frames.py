#!/usr/bin/env python3
"""
Dump frame-by-frame logs of ZI vs ZIC market execution.

Creates detailed log files showing:
- Each timestep's market state (bid, ask, spread)
- Each agent's action with their valuation/cost context
- Trade decisions with profitability analysis
- Cumulative efficiency tracking

Usage:
    python scripts/dump_zi_zic_frames.py

Output:
    results/zi_frames.log
    results/zic_frames.log
"""

import os
from typing import List, Dict, Any, Type
from dataclasses import dataclass

from engine.market import Market
from engine.orderbook import OrderBook
from engine.efficiency import (
    calculate_max_surplus,
    calculate_allocative_efficiency,
)
from traders.base import Agent
from traders.legacy.zi import ZI
from traders.legacy.zic import ZIC


# Multi-token setup (matching test_zi_zic_control.py)
BUYER_VALUATIONS = [
    [100, 90, 80, 70],  # Buyer 1
    [95, 85, 75, 65],   # Buyer 2
    [90, 80, 70, 60],   # Buyer 3
    [85, 75, 65, 55],   # Buyer 4
    [80, 70, 60, 50],   # Buyer 5
]

SELLER_COSTS = [
    [40, 50, 60, 70],   # Seller 1
    [45, 55, 65, 75],   # Seller 2
    [50, 60, 70, 80],   # Seller 3
    [55, 65, 75, 85],   # Seller 4
    [60, 70, 80, 90],   # Seller 5
]


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    time: int
    buyer_id: int
    seller_id: int
    price: int
    buyer_valuation: int
    seller_cost: int
    buyer_surplus: int
    seller_surplus: int
    total_surplus: int
    is_profitable: bool


def run_market_with_logging(
    agent_class: Type[Agent],
    agent_name: str,
    output_file: str,
    seed: int = 42,
    num_steps: int = 100,
) -> Dict[str, Any]:
    """
    Run a market and dump frame-by-frame logs.

    Args:
        agent_class: ZI or ZIC
        agent_name: "ZI" or "ZIC" for logging
        output_file: Path to output log file
        seed: Random seed
        num_steps: Number of timesteps

    Returns:
        Summary dict with efficiency metrics
    """
    num_tokens = 4
    max_surplus = calculate_max_surplus(BUYER_VALUATIONS, SELLER_COSTS)

    # Create agents
    buyers: List[Agent] = []
    for i, vals in enumerate(BUYER_VALUATIONS):
        agent = agent_class(
            player_id=i + 1,
            is_buyer=True,
            num_tokens=num_tokens,
            valuations=vals.copy(),
            price_min=0,
            price_max=150,
            seed=seed + i,
        )
        buyers.append(agent)

    sellers: List[Agent] = []
    for i, costs in enumerate(SELLER_COSTS):
        agent = agent_class(
            player_id=len(buyers) + i + 1,
            is_buyer=False,
            num_tokens=num_tokens,
            valuations=costs.copy(),
            price_min=0,
            price_max=150,
            seed=seed + len(buyers) + i,
        )
        sellers.append(agent)

    # Create market
    market = Market(
        num_buyers=len(buyers),
        num_sellers=len(sellers),
        price_min=0,
        price_max=150,
        num_times=num_steps,
        buyers=buyers,
        sellers=sellers,
        seed=seed + 100,
    )

    # Build valuation lookup
    buyer_vals: Dict[int, List[int]] = {i + 1: BUYER_VALUATIONS[i] for i in range(len(buyers))}
    seller_costs_lookup: Dict[int, List[int]] = {i + 1: SELLER_COSTS[i] for i in range(len(sellers))}

    # Track trades and surplus
    trades: List[TradeRecord] = []
    cumulative_surplus = 0

    # Open log file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"{agent_name} MARKET FRAME-BY-FRAME LOG\n")
        f.write(f"{'='*80}\n\n")

        f.write("MARKET SETUP:\n")
        f.write(f"  Agent Type: {agent_name}\n")
        f.write(f"  Buyers: {len(buyers)} (4 tokens each)\n")
        f.write(f"  Sellers: {len(sellers)} (4 tokens each)\n")
        f.write(f"  Max Surplus: {max_surplus}\n")
        f.write(f"  Timesteps: {num_steps}\n\n")

        f.write("BUYER VALUATIONS:\n")
        for i, vals in enumerate(BUYER_VALUATIONS):
            f.write(f"  Buyer {i+1}: {vals}\n")
        f.write("\n")

        f.write("SELLER COSTS:\n")
        for i, costs in enumerate(SELLER_COSTS):
            f.write(f"  Seller {i+1}: {costs}\n")
        f.write("\n")

        f.write(f"{'='*80}\n")
        f.write("TIMESTEP-BY-TIMESTEP EXECUTION\n")
        f.write(f"{'='*80}\n\n")

        # Run market timestep by timestep
        for t in range(1, num_steps + 1):
            market.run_time_step()
            ob = market.orderbook

            # Market state
            high_bid = int(ob.high_bid[t])
            low_ask = int(ob.low_ask[t])
            high_bidder = int(ob.high_bidder[t])
            low_asker = int(ob.low_asker[t])
            spread = (low_ask - high_bid) if (high_bid > 0 and low_ask > 0) else -1
            trade_price = int(ob.trade_price[t])
            trade_occurred = trade_price > 0

            f.write(f"=== TIMESTEP {t} ===\n")
            f.write(f"Market State: high_bid={high_bid}, low_ask={low_ask}, spread={spread}\n")

            # Agent actions - Buyers
            f.write("\nBuyer Actions:\n")
            for buyer_id in range(1, len(buyers) + 1):
                bid = int(ob.bids[buyer_id, t])
                prev_bid = int(ob.bids[buyer_id, t-1]) if t > 1 else 0
                buyer = buyers[buyer_id - 1]
                tokens_traded = buyer.num_trades
                current_valuation = buyer_vals[buyer_id][min(tokens_traded, len(buyer_vals[buyer_id])-1)]

                if bid > 0 and bid != prev_bid:
                    is_winner = (high_bidder == buyer_id)
                    status = "WINNER" if is_winner else "beaten"
                    # Check budget constraint
                    within_budget = bid <= current_valuation
                    budget_note = "" if within_budget else " [OVER BUDGET!]"
                    f.write(f"  Buyer {buyer_id}: bid @ {bid} → {status} "
                           f"(val={current_valuation}, token={tokens_traded+1}){budget_note}\n")
                else:
                    f.write(f"  Buyer {buyer_id}: pass (val={current_valuation}, token={tokens_traded+1})\n")

            # Agent actions - Sellers
            f.write("\nSeller Actions:\n")
            for seller_id in range(1, len(sellers) + 1):
                ask = int(ob.asks[seller_id, t])
                prev_ask = int(ob.asks[seller_id, t-1]) if t > 1 else 0
                seller = sellers[seller_id - 1]
                tokens_traded = seller.num_trades
                current_cost = seller_costs_lookup[seller_id][min(tokens_traded, len(seller_costs_lookup[seller_id])-1)]

                if ask > 0 and ask != prev_ask:
                    is_winner = (low_asker == seller_id)
                    status = "WINNER" if is_winner else "beaten"
                    # Check budget constraint
                    within_budget = ask >= current_cost
                    budget_note = "" if within_budget else " [BELOW COST!]"
                    f.write(f"  Seller {seller_id}: ask @ {ask} → {status} "
                           f"(cost={current_cost}, token={tokens_traded+1}){budget_note}\n")
                else:
                    f.write(f"  Seller {seller_id}: pass (cost={current_cost}, token={tokens_traded+1})\n")

            # Trade decision
            f.write("\nTrade Decision:\n")
            if high_bidder > 0 and low_asker > 0:
                buyer = buyers[high_bidder - 1]
                seller = sellers[low_asker - 1]

                # Get valuations BEFORE this trade happened
                buyer_token_idx = buyer.num_trades - (1 if trade_occurred else 0)
                seller_token_idx = seller.num_trades - (1 if trade_occurred else 0)
                buyer_val = buyer_vals[high_bidder][max(0, buyer_token_idx)]
                seller_cost = seller_costs_lookup[low_asker][max(0, seller_token_idx)]

                buyer_accepted = bool(ob.buyer_accepted[t])
                seller_accepted = bool(ob.seller_accepted[t])

                # Profitability analysis
                buyer_profitable = low_ask < buyer_val if low_ask > 0 else False
                seller_profitable = high_bid > seller_cost if high_bid > 0 else False

                f.write(f"  High Bidder (Buyer {high_bidder}): "
                       f"{'ACCEPT' if buyer_accepted else 'REJECT'} "
                       f"(ask={low_ask} vs val={buyer_val}, "
                       f"{'profitable' if buyer_profitable else 'UNPROFITABLE'})\n")
                f.write(f"  Low Asker (Seller {low_asker}): "
                       f"{'ACCEPT' if seller_accepted else 'REJECT'} "
                       f"(bid={high_bid} vs cost={seller_cost}, "
                       f"{'profitable' if seller_profitable else 'UNPROFITABLE'})\n")

                if trade_occurred:
                    buyer_surplus = buyer_val - trade_price
                    seller_surplus = trade_price - seller_cost
                    total_surplus = buyer_surplus + seller_surplus
                    is_profitable_trade = buyer_val > seller_cost

                    cumulative_surplus += total_surplus
                    efficiency = calculate_allocative_efficiency(cumulative_surplus, max_surplus)

                    trades.append(TradeRecord(
                        time=t,
                        buyer_id=high_bidder,
                        seller_id=low_asker,
                        price=trade_price,
                        buyer_valuation=buyer_val,
                        seller_cost=seller_cost,
                        buyer_surplus=buyer_surplus,
                        seller_surplus=seller_surplus,
                        total_surplus=total_surplus,
                        is_profitable=is_profitable_trade,
                    ))

                    f.write(f"\n  → TRADE @ {trade_price}!\n")
                    f.write(f"     Buyer surplus: {buyer_val} - {trade_price} = {buyer_surplus}\n")
                    f.write(f"     Seller surplus: {trade_price} - {seller_cost} = {seller_surplus}\n")
                    f.write(f"     Total surplus: {total_surplus} "
                           f"({'PROFITABLE' if is_profitable_trade else 'UNPROFITABLE TRADE!'})\n")
                else:
                    efficiency = calculate_allocative_efficiency(cumulative_surplus, max_surplus) if cumulative_surplus > 0 else 0
                    f.write(f"\n  → NO TRADE (spread not crossed or rejected)\n")
            else:
                efficiency = calculate_allocative_efficiency(cumulative_surplus, max_surplus) if cumulative_surplus > 0 else 0
                f.write(f"  No standing bid or ask\n")

            # Running totals
            profitable_trades = sum(1 for tr in trades if tr.is_profitable)
            unprofitable_trades = len(trades) - profitable_trades
            f.write(f"\nRunning Totals:\n")
            f.write(f"  Trades: {len(trades)} ({profitable_trades} profitable, {unprofitable_trades} unprofitable)\n")
            f.write(f"  Cumulative Surplus: {cumulative_surplus}\n")
            f.write(f"  Max Surplus: {max_surplus}\n")
            f.write(f"  Efficiency: {efficiency:.1f}%\n")
            f.write("\n")

        # Final summary
        f.write(f"{'='*80}\n")
        f.write("FINAL SUMMARY\n")
        f.write(f"{'='*80}\n\n")

        profitable_trades = sum(1 for tr in trades if tr.is_profitable)
        unprofitable_trades = len(trades) - profitable_trades
        final_efficiency = calculate_allocative_efficiency(cumulative_surplus, max_surplus)

        f.write(f"Total Trades: {len(trades)}\n")
        f.write(f"  Profitable: {profitable_trades}\n")
        f.write(f"  Unprofitable: {unprofitable_trades}\n")
        f.write(f"\nFinal Surplus: {cumulative_surplus}\n")
        f.write(f"Max Surplus: {max_surplus}\n")
        f.write(f"Final Efficiency: {final_efficiency:.1f}%\n\n")

        if trades:
            f.write("ALL TRADES:\n")
            for i, tr in enumerate(trades, 1):
                profit_marker = "✓" if tr.is_profitable else "✗"
                f.write(f"  {i}. t={tr.time}: Buyer {tr.buyer_id} (v={tr.buyer_valuation}) ↔ "
                       f"Seller {tr.seller_id} (c={tr.seller_cost}) @ {tr.price} "
                       f"[surplus={tr.total_surplus}] {profit_marker}\n")

    return {
        "agent_type": agent_name,
        "total_trades": len(trades),
        "profitable_trades": profitable_trades,
        "unprofitable_trades": unprofitable_trades,
        "final_surplus": cumulative_surplus,
        "max_surplus": max_surplus,
        "efficiency": final_efficiency,
    }


def main() -> None:
    """Run ZI and ZIC markets and dump frame logs."""
    print("Dumping ZI vs ZIC frame-by-frame logs...")
    print()

    # Run ZI
    print("Running ZI market...")
    zi_results = run_market_with_logging(
        agent_class=ZI,
        agent_name="ZI",
        output_file="results/zi_frames.log",
        seed=42,
    )
    print(f"  Written: results/zi_frames.log")
    print(f"  Efficiency: {zi_results['efficiency']:.1f}%")
    print(f"  Trades: {zi_results['total_trades']} "
          f"({zi_results['profitable_trades']} profitable, "
          f"{zi_results['unprofitable_trades']} unprofitable)")
    print()

    # Run ZIC
    print("Running ZIC market...")
    zic_results = run_market_with_logging(
        agent_class=ZIC,
        agent_name="ZIC",
        output_file="results/zic_frames.log",
        seed=42,
    )
    print(f"  Written: results/zic_frames.log")
    print(f"  Efficiency: {zic_results['efficiency']:.1f}%")
    print(f"  Trades: {zic_results['total_trades']} "
          f"({zic_results['profitable_trades']} profitable, "
          f"{zic_results['unprofitable_trades']} unprofitable)")
    print()

    # Summary comparison
    print("=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'ZI':>15} {'ZIC':>15}")
    print("-" * 60)
    print(f"{'Efficiency':<25} {zi_results['efficiency']:>14.1f}% {zic_results['efficiency']:>14.1f}%")
    print(f"{'Total Trades':<25} {zi_results['total_trades']:>15} {zic_results['total_trades']:>15}")
    print(f"{'Profitable Trades':<25} {zi_results['profitable_trades']:>15} {zic_results['profitable_trades']:>15}")
    print(f"{'Unprofitable Trades':<25} {zi_results['unprofitable_trades']:>15} {zic_results['unprofitable_trades']:>15}")
    print(f"{'Difference':<25} {zic_results['efficiency'] - zi_results['efficiency']:>14.1f}pp")
    print("=" * 60)


if __name__ == "__main__":
    main()
