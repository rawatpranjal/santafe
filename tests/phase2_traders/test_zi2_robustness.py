"""
Robustness Tests for ZI2 - Stress testing across market conditions.

Tests ZI2's stability across different market sizes, token counts, and price ranges.
"""

import pytest
import numpy as np
from traders.legacy.zi2 import ZI2
from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
)


def run_zi2_market(num_agents, num_tokens, price_range, seed=42):
    """Helper to run ZI2 market and return efficiency."""
    price_min, price_max = price_range

    # Generate valuations/costs with proper range
    buyer_vals = [
        sorted(np.random.randint(price_max//2, price_max, num_tokens).tolist(), reverse=True)
        for _ in range(num_agents)
    ]
    seller_costs = [
        sorted(np.random.randint(price_min, price_max//2, num_tokens).tolist())
        for _ in range(num_agents)
    ]

    buyers = [
        ZI2(i+1, True, num_tokens, buyer_vals[i], price_min, price_max, seed=seed+i)
        for i in range(num_agents)
    ]
    sellers = [
        ZI2(i+1+num_agents, False, num_tokens, seller_costs[i], price_min, price_max, seed=seed+i+100)
        for i in range(num_agents)
    ]

    market = Market(
        num_buyers=num_agents,
        num_sellers=num_agents,
        num_times=max(100, num_tokens * 30),  # Scale time with tokens
        price_min=price_min,
        price_max=price_max,
        buyers=buyers,
        sellers=sellers,
        seed=seed
    )

    for _ in range(market.num_times):
        if not market.run_time_step():
            break

    trades = extract_trades_from_orderbook(market.orderbook, market.num_times)
    buyer_valuations = {i+1: buyers[i].valuations for i in range(num_agents)}
    seller_valuations = {i+1: sellers[i].valuations for i in range(num_agents)}

    actual = calculate_actual_surplus(trades, buyer_valuations, seller_valuations)
    max_surplus = calculate_max_surplus(
        [b.valuations for b in buyers],
        [s.valuations for s in sellers]
    )

    if max_surplus > 0:
        return calculate_allocative_efficiency(actual, max_surplus)
    return 0.0


def test_zi2_different_market_sizes():
    """Test ZI2 across different market sizes (2v2, 4v4, 8v8)."""
    market_sizes = [2, 4, 8]
    results = {}

    for size in market_sizes:
        efficiencies = []
        for rep in range(5):
            eff = run_zi2_market(
                num_agents=size,
                num_tokens=3,
                price_range=(0, 200),
                seed=rep*1000
            )
            if eff > 0:
                efficiencies.append(eff)

        if efficiencies:
            avg_eff = np.mean(efficiencies)
            results[size] = avg_eff
            print(f"  {size}v{size}: {avg_eff:.2f}%")

    print(f"\nMarket Size Robustness:")
    for size, eff in results.items():
        print(f"  {size}v{size}: {eff:.2f}%")

    # All sizes should achieve reasonable efficiency (>85%)
    for size, eff in results.items():
        assert eff > 85, f"{size}v{size} market efficiency {eff:.2f}% below 85%"


def test_zi2_different_token_counts():
    """Test ZI2 with varying token counts (1, 3, 5, 10)."""
    token_counts = [1, 3, 5, 10]
    results = {}

    for num_tokens in token_counts:
        efficiencies = []
        for rep in range(5):
            eff = run_zi2_market(
                num_agents=4,
                num_tokens=num_tokens,
                price_range=(0, 200),
                seed=rep*1000
            )
            if eff > 0:
                efficiencies.append(eff)

        if efficiencies:
            avg_eff = np.mean(efficiencies)
            results[num_tokens] = avg_eff

    print(f"\nToken Count Robustness:")
    for tokens, eff in results.items():
        print(f"  {tokens} tokens: {eff:.2f}%")

    # All token counts should achieve reasonable efficiency
    for tokens, eff in results.items():
        assert eff > 85, f"{tokens} tokens: efficiency {eff:.2f}% below 85%"


def test_zi2_tight_equilibrium():
    """Test ZI2 with tight equilibrium (narrow price range)."""
    # Tight range [200, 210] - only 10 price levels
    efficiencies = []

    for rep in range(10):
        buyers = [
            ZI2(i+1, True, 3, [210, 208, 206], 200, 210, seed=rep*100+i)
            for i in range(4)
        ]
        sellers = [
            ZI2(i+5, False, 3, [200, 202, 204], 200, 210, seed=rep*100+i+4)
            for i in range(4)
        ]

        market = Market(
            num_buyers=4,
            num_sellers=4,
            num_times=100,
            price_min=200,
            price_max=210,
            buyers=buyers,
            sellers=sellers,
            seed=rep
        )

        for _ in range(100):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, 100)
        buyer_vals = {i+1: buyers[i].valuations for i in range(4)}
        seller_costs = {i+1: sellers[i].valuations for i in range(4)}

        actual = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            eff = calculate_allocative_efficiency(actual, max_surplus)
            efficiencies.append(eff)

    avg_eff = np.mean(efficiencies)
    print(f"\nTight Equilibrium Test:")
    print(f"  Price range: [200, 210] (narrow)")
    print(f"  Efficiency: {avg_eff:.2f}%")

    # Should still achieve reasonable efficiency even in tight range
    assert avg_eff > 75, f"Tight equilibrium efficiency {avg_eff:.2f}% below 75%"


def test_zi2_asymmetric_markets():
    """Test ZI2 with unequal supply/demand (6 buyers vs 4 sellers)."""
    efficiencies = []

    for rep in range(10):
        # 6 buyers, 4 sellers (excess demand)
        buyers = [
            ZI2(i+1, True, 3, [200, 180, 160], 0, 250, seed=rep*100+i)
            for i in range(6)
        ]
        sellers = [
            ZI2(i+7, False, 3, [40, 60, 80], 0, 250, seed=rep*100+i+6)
            for i in range(4)
        ]

        market = Market(
            num_buyers=6,
            num_sellers=4,
            num_times=150,
            price_min=0,
            price_max=250,
            buyers=buyers,
            sellers=sellers,
            seed=rep
        )

        for _ in range(150):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, 150)
        buyer_vals = {i+1: buyers[i].valuations for i in range(6)}
        seller_costs = {i+1: sellers[i].valuations for i in range(4)}

        actual = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            eff = calculate_allocative_efficiency(actual, max_surplus)
            efficiencies.append(eff)

    avg_eff = np.mean(efficiencies)
    print(f"\nAsymmetric Market Test:")
    print(f"  Setup: 6 buyers vs 4 sellers")
    print(f"  Efficiency: {avg_eff:.2f}%")

    # Should still achieve reasonable efficiency with asymmetry
    assert avg_eff > 80, f"Asymmetric market efficiency {avg_eff:.2f}% below 80%"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
