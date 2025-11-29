"""
Self-Play Efficiency Test for Perry Trader.

Tests Perry vs Perry homogeneous market to validate statistical prediction strategy.
Perry ranked 6th in 1993 tournament, expect 85-92% self-play efficiency.
"""

import numpy as np
from traders.legacy.perry import Perry
from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
)


def test_perry_self_play_efficiency():
    """
    Test Perry vs Perry homogeneous market efficiency.

    Perry's statistical prediction strategy should achieve high efficiency
    when all agents use the same approach. Expected 85-92% based on:
    - 6th place finish in 1993 tournament (strong performer)
    - Statistical prediction with adaptive learning (should converge)
    - Better than ZIP (85%) due to efficiency-based tuning
    - Close to GD (97%) but without optimal belief-based pricing

    Methodology:
    - 5 Perry buyers vs 5 Perry sellers
    - 10 replications with different random seeds
    - Large token sets to maximize trading opportunities
    - 200 time steps per market (ample time for convergence)
    """
    num_agents = 5
    num_tokens = 5

    # Diverse token valuations to test various scenarios
    buyer_tokens = [
        [200, 180, 160, 140, 120],
        [195, 175, 155, 135, 115],
        [190, 170, 150, 130, 110],
        [185, 165, 145, 125, 105],
        [180, 160, 140, 120, 100],
    ]

    seller_tokens = [
        [20, 40, 60, 80, 100],
        [25, 45, 65, 85, 105],
        [30, 50, 70, 90, 110],
        [35, 55, 75, 95, 115],
        [40, 60, 80, 100, 120],
    ]

    efficiencies = []

    # Run 10 replications
    for rep in range(10):
        buyers = [
            Perry(
                player_id=i+1,
                is_buyer=True,
                num_tokens=num_tokens,
                valuations=buyer_tokens[i],
                price_min=0,
                price_max=250,
                num_buyers=num_agents,
                num_sellers=num_agents,
                num_times=200,
                seed=rep*100+i
            )
            for i in range(num_agents)
        ]

        sellers = [
            Perry(
                player_id=i+1,
                is_buyer=False,
                num_tokens=num_tokens,
                valuations=seller_tokens[i],
                price_min=0,
                price_max=250,
                num_buyers=num_agents,
                num_sellers=num_agents,
                num_times=200,
                seed=rep*100+i+num_agents
            )
            for i in range(num_agents)
        ]

        market = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=200,
            price_min=0,
            price_max=250,
            buyers=buyers,
            sellers=sellers,
            seed=rep
        )

        # Run market
        for _ in range(200):
            if not market.run_time_step():
                break

        # Calculate efficiency
        trades = extract_trades_from_orderbook(market.orderbook, 200)
        buyer_valuations = {i+1: buyers[i].valuations for i in range(num_agents)}
        seller_costs = {i+1: sellers[i].valuations for i in range(num_agents)}

        actual_surplus = calculate_actual_surplus(trades, buyer_valuations, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

    # Analyze results
    avg_efficiency = np.mean(efficiencies)
    std_efficiency = np.std(efficiencies)
    min_efficiency = np.min(efficiencies)
    max_efficiency = np.max(efficiencies)

    print(f"\nPerry Self-Play Efficiency Results:")
    print(f"Average: {avg_efficiency:.2f}% ± {std_efficiency:.2f}%")
    print(f"Range: [{min_efficiency:.2f}%, {max_efficiency:.2f}%]")
    print(f"Replications: {len(efficiencies)}")
    print(f"\nComparison to Other Traders:")
    print(f"  Lin: 99.85% (BEST)")
    print(f"  ZIC: ~98%")
    print(f"  GD: ~97%")
    print(f"  Perry: {avg_efficiency:.2f}% ← This result")
    print(f"  ZIP: ~85%")

    # Validation criteria based on 6th place finish
    assert avg_efficiency >= 80.0, \
        f"Perry self-play efficiency {avg_efficiency:.2f}% below minimum 80%"

    assert avg_efficiency >= 85.0, \
        f"Perry self-play efficiency {avg_efficiency:.2f}% below expected 85% (6th place should beat ZIP's 85%)"

    # Perry should not exceed GD's 97% (GD is belief-based optimal)
    if avg_efficiency > 98.0:
        print(f"⚠️  WARNING: Perry efficiency {avg_efficiency:.2f}% unexpectedly high (exceeds GD's 97%)")

    print(f"\n✓ Perry self-play validation PASSED")
    print(f"  Statistical prediction strategy achieves {avg_efficiency:.2f}% efficiency")
    print(f"  Validates 6th place finish in 1993 tournament")


if __name__ == "__main__":
    test_perry_self_play_efficiency()
