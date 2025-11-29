#!/usr/bin/env python3
"""
Debug ZIP behavior to understand why efficiency is low.
"""

import numpy as np
from engine.market import Market
from traders.legacy.zip import ZIP


def debug_single_market():
    """Run one market and trace ZIP behavior."""
    
    # Simple symmetric market
    buyer_vals = [[300], [275], [250], [225]]
    seller_costs = [[100], [125], [150], [175]]
    
    num_buyers = len(buyer_vals)
    num_sellers = len(seller_costs)
    
    # Create agents
    agents = []
    for i, vals in enumerate(buyer_vals):
        agent = ZIP(
            player_id=i + 1,
            is_buyer=True,
            num_tokens=len(vals),
            valuations=vals,
            num_times=200,
            price_min=0,
            price_max=400
        )
        agents.append(agent)
        
    for i, costs in enumerate(seller_costs):
        agent = ZIP(
            player_id=num_buyers + i + 1,
            is_buyer=False,
            num_tokens=len(costs),
            valuations=costs,
            num_times=200,
            price_min=0,
            price_max=400
        )
        agents.append(agent)
    
    # Run market with tracing
    for a in agents:
        a.start_period(1)
        print(f"Agent {a.player_id} ({'B' if a.is_buyer else 'S'}): limit={a.valuations[0]}, margin={a.margin:.3f}, beta={a.beta:.3f}, gamma={a.gamma:.3f}")
        
    market = Market(
        num_buyers=num_buyers,
        num_sellers=num_sellers,
        price_min=0,
        price_max=400,
        num_times=200,
        buyers=[a for a in agents if a.is_buyer],
        sellers=[a for a in agents if not a.is_buyer]
    )
    
    trade_count = 0
    for step in range(20):  # First 20 steps only
        t = market.current_time
        market.run_time_step()
        
        # Check if trade occurred
        if market.orderbook.trade_price[t+1] > 0:
            trade_count += 1
            price = int(market.orderbook.trade_price[t+1])
            print(f"\nStep {t+1}: TRADE at ${price}")
            
            # Print all agent margins after trade
            for a in agents:
                quote = a._calculate_quote()
                print(f"  Agent {a.player_id} ({'B' if a.is_buyer else 'S'}): margin={a.margin:.3f}, quote={quote}")
        else:
            # Print current book state
            hb = int(market.orderbook.high_bid[t+1])
            la = int(market.orderbook.low_ask[t+1])
            if hb > 0 or la > 0:
                print(f"Step {t+1}: Book: bid={hb}, ask={la}")
    
    for a in agents:
        a.end_period()
    
    print(f"\nTotal Trades: {trade_count}")
    print(f"Total Profit: {sum(a.period_profit for a in agents)}")
    
    # Max surplus
    all_vals = sorted([v for vals in buyer_vals for v in vals], reverse=True)
    all_costs = sorted([c for costs in seller_costs for c in costs])
    max_surplus = sum(v - c for v, c in zip(all_vals, all_costs) if v > c)
    print(f"Max Surplus: {max_surplus}")
    print(f"Efficiency: {sum(a.period_profit for a in agents) / max_surplus * 100:.1f}%")


if __name__ == "__main__":
    debug_single_market()
