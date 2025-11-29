"""
Diagnostic script to trace GD vs ZIC interaction issues.

Runs a single period with detailed logging of:
- GD's belief calculations
- GD's expected surplus calculations
- Quote prices and acceptance decisions
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.market import Market
from engine.token_generator import TokenGenerator
from traders.legacy.zic import ZIC
from traders.legacy.gd import GD
import numpy as np

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s'
)

def main():
    print("=" * 80)
    print("DIAGNOSTIC: GD vs ZIC Interaction Analysis")
    print("=" * 80)

    # Create market with 3 GD + 2 ZIC on each side
    num_buyers = 5
    num_sellers = 5
    num_tokens = 4
    num_periods = 1  # Just one period for detailed analysis
    num_steps = 100
    price_min = 1
    price_max = 1000

    # Generate valuations
    seed = 42
    token_gen = TokenGenerator(
        game_type=6453,
        num_tokens=num_tokens,
        seed=seed
    )

    # Initialize round parameters
    token_gen.new_round()

    # Generate tokens for each agent
    buyer_values = []
    seller_costs = []
    for i in range(num_buyers):
        buyer_values.append(token_gen.generate_tokens(is_buyer=True))
    for i in range(num_sellers):
        seller_costs.append(token_gen.generate_tokens(is_buyer=False))

    print("\nValuations:")
    for i in range(num_buyers):
        print(f"  Buyer {i}: {buyer_values[i]}")
    for i in range(num_sellers):
        print(f"  Seller {i}: {seller_costs[i]}")

    # Create agents: 3 GD + 2 ZIC
    buyers = []
    sellers = []

    buyer_types = ["GD", "GD", "GD", "ZIC", "ZIC"]
    seller_types = ["GD", "GD", "GD", "ZIC", "ZIC"]

    for i in range(num_buyers):
        agent_type = buyer_types[i]
        player_id = i + 1  # Player IDs start from 1
        if agent_type == "GD":
            agent = GD(
                player_id=player_id,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=buyer_values[i],
                price_min=price_min,
                price_max=price_max,
                memory_length=8,
                seed=seed + i
            )
        else:  # ZIC
            agent = ZIC(
                player_id=player_id,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=buyer_values[i],
                price_min=price_min,
                price_max=price_max,
                seed=seed + i
            )
        buyers.append(agent)

    for i in range(num_sellers):
        agent_type = seller_types[i]
        player_id = i + 1  # Player IDs start from 1
        if agent_type == "GD":
            agent = GD(
                player_id=player_id,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=seller_costs[i],
                price_min=price_min,
                price_max=price_max,
                memory_length=8,
                seed=seed + i + num_buyers
            )
        else:  # ZIC
            agent = ZIC(
                player_id=player_id,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=seller_costs[i],
                price_min=price_min,
                price_max=price_max,
                seed=seed + i + num_buyers
            )
        sellers.append(agent)

    print("\nAgents:")
    for i, agent in enumerate(buyers):
        print(f"  Buyer {i}: {type(agent).__name__}")
    for i, agent in enumerate(sellers):
        print(f"  Seller {i}: {type(agent).__name__}")

    # Create market
    market = Market(
        buyers=buyers,
        sellers=sellers,
        num_periods=num_periods,
        num_steps=num_steps,
        price_min=price_min,
        price_max=price_max,
        seed=seed
    )

    # Run one period with detailed tracing
    print("\n" + "=" * 80)
    print("RUNNING PERIOD 1")
    print("=" * 80)

    results = market.run_period()

    print("\n" + "=" * 80)
    print("PERIOD RESULTS")
    print("=" * 80)
    print(f"Efficiency: {results['efficiency']:.2f}%")
    print(f"Actual Surplus: {results['actual_surplus']}")
    print(f"Max Surplus: {results['max_surplus']}")
    print(f"Num Trades: {len(results['trades'])}")

    print("\nTrades:")
    for trade in results['trades']:
        print(f"  Step {trade['time']}: Price {trade['price']}, "
              f"Buyer {trade['buyer_id']}, Seller {trade['seller_id']}")

    # Now analyze GD agent histories
    print("\n" + "=" * 80)
    print("GD AGENT ANALYSIS")
    print("=" * 80)

    for i, agent in enumerate(buyers[:3]):  # First 3 are GD
        if isinstance(agent, GD):
            print(f"\nGD Buyer {i}:")
            print(f"  History size: {len(agent.history)}")
            print(f"  Last 10 history entries:")
            for price, is_bid, accepted in agent.history[-10:]:
                side = "BID" if is_bid else "ASK"
                status = "ACCEPTED" if accepted else "REJECTED"
                print(f"    {side} @ {price}: {status}")

            # Check belief at their current valuation
            if agent.num_trades < len(agent.valuations):
                val = agent.valuations[agent.num_trades]
                belief = agent._belief_bid_accepted(val)
                print(f"  Belief at valuation {val}: {belief:.3f}")

    for i, agent in enumerate(sellers[:3]):  # First 3 are GD
        if isinstance(agent, GD):
            print(f"\nGD Seller {i}:")
            print(f"  History size: {len(agent.history)}")
            print(f"  Last 10 history entries:")
            for price, is_bid, accepted in agent.history[-10:]:
                side = "BID" if is_bid else "ASK"
                status = "ACCEPTED" if accepted else "REJECTED"
                print(f"    {side} @ {price}: {status}")

            # Check belief at their current cost
            if agent.num_trades < len(agent.valuations):
                cost = agent.valuations[agent.num_trades]
                belief = agent._belief_ask_accepted(cost)
                print(f"  Belief at cost {cost}: {belief:.3f}")

if __name__ == "__main__":
    main()
