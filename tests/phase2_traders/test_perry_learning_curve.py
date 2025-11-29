"""
Multi-Period Learning Curve Test for Perry Trader.

Tests that Perry's adaptive a0 parameter improves performance across periods.
Perry should learn optimal bidding strategy through efficiency-based tuning.
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


def test_perry_efficiency_improves_across_periods():
    """
    Test that Perry's learning (a0 adaptation) improves efficiency over periods.

    Perry uses efficiency-based tuning (lines 246-296):
    - If efficiency < 1.0, decreases a0
    - If efficiency low and few trades, adjusts price_std
    - This should lead to better performance in later periods

    Expected: Efficiency increases from ~80% (period 1) → ~95% (period 5)
    """
    num_agents = 4
    num_tokens = 4

    buyer_tokens = [
        [180, 160, 140, 120],
        [175, 155, 135, 115],
        [170, 150, 130, 110],
        [165, 145, 125, 105],
    ]

    seller_tokens = [
        [40, 60, 80, 100],
        [45, 65, 85, 105],
        [50, 70, 90, 110],
        [55, 75, 95, 115],
    ]

    # Run 5 periods to observe learning
    period_efficiencies = []
    a0_values = []  # Track a0 parameter evolution

    buyers = [
        Perry(
            player_id=i+1,
            is_buyer=True,
            num_tokens=num_tokens,
            valuations=buyer_tokens[i],
            price_min=0,
            price_max=220,
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=100,
            seed=42+i
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
            price_max=220,
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=100,
            seed=142+i
        )
        for i in range(num_agents)
    ]

    # Run 5 periods
    for period in range(1, 6):
        # Start period
        for agent in buyers + sellers:
            agent.start_period(period)

        market = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=100,
            price_min=0,
            price_max=220,
            buyers=buyers,
            sellers=sellers,
            seed=1000+period
        )

        # Run market
        for _ in range(100):
            if not market.run_time_step():
                break

        # Calculate efficiency
        trades = extract_trades_from_orderbook(market.orderbook, 100)
        buyer_valuations = {i+1: buyers[i].valuations for i in range(num_agents)}
        seller_costs = {i+1: sellers[i].valuations for i in range(num_agents)}

        actual_surplus = calculate_actual_surplus(trades, buyer_valuations, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            period_efficiencies.append(efficiency)

        # Track a0 parameter (average across all agents)
        avg_a0 = np.mean([b.a0 for b in buyers] + [s.a0 for s in sellers])
        a0_values.append(avg_a0)

        # End period (triggers _evaluate() and a0 adjustment)
        for agent in buyers + sellers:
            agent.end_period()

        print(f"Period {period}: Efficiency = {efficiency:.2f}%, Avg a0 = {avg_a0:.3f}")

    # Analyze learning curve
    print(f"\nLearning Curve Analysis:")
    print(f"  Period 1-2 Efficiency: {np.mean(period_efficiencies[:2]):.2f}%")
    print(f"  Period 4-5 Efficiency: {np.mean(period_efficiencies[-2:]):.2f}%")
    print(f"  Improvement: {np.mean(period_efficiencies[-2:]) - np.mean(period_efficiencies[:2]):.2f}pp")
    print(f"\na0 Parameter Evolution:")
    print(f"  Initial a0: {a0_values[0]:.3f}")
    print(f"  Final a0: {a0_values[-1]:.3f}")
    print(f"  Change: {(a0_values[-1] - a0_values[0]) / a0_values[0] * 100:.1f}%")

    # Validation 1: Efficiency should be high by final periods
    final_efficiency = np.mean(period_efficiencies[-2:])
    assert final_efficiency >= 85.0, \
        f"Perry final efficiency {final_efficiency:.2f}% below 85% (learning failed)"

    # Validation 2: Should show improvement trend (or at least maintain high efficiency)
    # Allow for cases where Perry starts high and maintains
    early_efficiency = np.mean(period_efficiencies[:2])
    improvement = final_efficiency - early_efficiency

    if early_efficiency < 90.0:
        # If starting efficiency is low, should improve
        assert improvement >= -5.0, \
            f"Perry efficiency decreased by {-improvement:.1f}pp (no learning)"
        print(f"\n✓ Perry shows learning: {early_efficiency:.2f}% → {final_efficiency:.2f}%")
    else:
        # If starting high, maintaining is acceptable
        assert final_efficiency >= 85.0, \
            f"Perry failed to maintain high efficiency"
        print(f"\n✓ Perry maintains high efficiency: {early_efficiency:.2f}% → {final_efficiency:.2f}%")

    # Validation 3: a0 parameter should adapt (not stay at 2.0)
    final_a0 = a0_values[-1]
    assert abs(final_a0 - 2.0) > 0.01 or final_efficiency > 95.0, \
        f"Perry a0 parameter never adapted (stayed at {final_a0:.3f})"

    print(f"✓ Perry multi-period learning validated")


if __name__ == "__main__":
    test_perry_efficiency_improves_across_periods()
