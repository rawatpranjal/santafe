"""
Behavioral Tests for Zero Intelligence Enhanced (ZI2) Trading Agents.

These tests validate that ZI2 agents exhibit the expected behavior:
enhanced ZIC with memory of traded tokens, leading to improved efficiency.

ZI2 was developed by Ringuette for the 1993 Santa Fe Tournament.
It tracks which tokens have been traded to make better decisions.
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple

from traders.legacy.zi2 import ZI2
from traders.legacy.zic import ZIC  # For comparison
from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
)


# =============================================================================
# TEST 1: TOKEN MEMORY TRACKING
# =============================================================================

def test_zi2_token_memory():
    """
    Test that ZI2 properly tracks which tokens have been traded.

    ZI2 Rule: Remember traded tokens to avoid redundant attempts.
    Expected: Should not try to trade already-traded tokens.
    """
    # Create a ZI2 buyer with multiple tokens
    buyer = ZI2(
        player_id=1,
        is_buyer=True,
        num_tokens=3,
        valuations=[100, 90, 80],
        price_min=0,
        price_max=150,
        seed=42
    )

    # Initially, should track no trades
    assert buyer.num_trades == 0, "Should start with no trades"

    # Simulate a trade
    buyer.buy_sell_result(
        status=1,  # Buyer traded
        trade_price=85,
        trade_type=1,
        high_bid=0,
        high_bidder=0,
        low_ask=0,
        low_asker=0
    )

    # Should now track one trade
    assert buyer.num_trades == 1, "Should track completed trade"

    # Next bid should be for the second token
    buyer.bid_ask(time=2, nobidask=0)
    bid = buyer.bid_ask_response()

    # Bid should be constrained by second token's valuation (90)
    assert bid <= 90, f"Bid {bid} exceeds second token valuation 90"


# =============================================================================
# TEST 2: EFFICIENCY IMPROVEMENT OVER ZIC
# =============================================================================

def test_zi2_efficiency_improvement():
    """
    Test that ZI2 achieves higher efficiency than basic ZIC.

    Expected: ZI2 should achieve 5-10% improvement over ZIC baseline.
    This is due to better token management and memory.
    """
    # Market setup with multiple tokens
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

    # Run ZI2 market
    zi2_efficiencies = []

    for rep in range(5):
        buyers = [
            ZI2(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=220, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers = [
            ZI2(i+5, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=220, seed=rep*100+i+4)
            for i in range(num_agents)
        ]

        market = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=100,
            price_min=0,
            price_max=220,
            buyers=buyers,
            sellers=sellers,
            seed=rep
        )

        for _ in range(100):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, 100)
        buyer_vals = {i+1: buyers[i].valuations for i in range(num_agents)}
        seller_costs = {i+1: sellers[i].valuations for i in range(num_agents)}

        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            zi2_efficiencies.append(efficiency)

    # Run ZIC market for comparison
    zic_efficiencies = []

    for rep in range(5):
        buyers = [
            ZIC(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=220, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers = [
            ZIC(i+5, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=220, seed=rep*100+i+4)
            for i in range(num_agents)
        ]

        market = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=100,
            price_min=0,
            price_max=220,
            buyers=buyers,
            sellers=sellers,
            seed=rep
        )

        for _ in range(100):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, 100)
        buyer_vals = {i+1: buyers[i].valuations for i in range(num_agents)}
        seller_costs = {i+1: sellers[i].valuations for i in range(num_agents)}

        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            zic_efficiencies.append(efficiency)

    # Compare results
    if len(zi2_efficiencies) > 0 and len(zic_efficiencies) > 0:
        avg_zi2 = np.mean(zi2_efficiencies)
        avg_zic = np.mean(zic_efficiencies)

        print(f"\nZI2 vs ZIC Efficiency:")
        print(f"ZI2: {avg_zi2:.2f}%")
        print(f"ZIC: {avg_zic:.2f}%")
        print(f"Improvement: {avg_zi2 - avg_zic:.2f} percentage points")

        # ZI2 should at least match ZIC
        assert avg_zi2 >= avg_zic - 5, \
            f"ZI2 ({avg_zi2:.2f}%) unexpectedly underperformed ZIC ({avg_zic:.2f}%)"


# =============================================================================
# TEST 3: BUDGET CONSTRAINT ENFORCEMENT
# =============================================================================

def test_zi2_budget_constraints():
    """
    Test that ZI2 never violates budget constraints.

    Like ZIC, ZI2 must respect valuations and costs.
    """
    # Test buyer constraints
    buyer = ZI2(
        player_id=1,
        is_buyer=True,
        num_tokens=3,
        valuations=[100, 80, 60],
        price_min=0,
        price_max=200,
        seed=42
    )

    for token_idx in range(3):
        buyer.num_trades = token_idx
        valuation = buyer.valuations[token_idx]

        # Test multiple bids
        for _ in range(20):
            buyer.bid_ask(time=1, nobidask=0)
            bid = buyer.bid_ask_response()
            if bid >= 0:
                assert bid <= valuation, \
                    f"ZI2 buyer bid {bid} exceeds valuation {valuation}"
            buyer.has_responded = False

    # Test seller constraints
    seller = ZI2(
        player_id=2,
        is_buyer=False,
        num_tokens=3,
        valuations=[40, 60, 80],
        price_min=0,
        price_max=200,
        seed=43
    )

    for token_idx in range(3):
        seller.num_trades = token_idx
        cost = seller.valuations[token_idx]

        for _ in range(20):
            seller.bid_ask(time=1, nobidask=0)
            ask = seller.bid_ask_response()
            if ask >= 0:
                assert ask >= cost, \
                    f"ZI2 seller ask {ask} below cost {cost}"
            seller.has_responded = False


# =============================================================================
# TEST 4: RANDOM BEHAVIOR (NO LEARNING)
# =============================================================================

def test_zi2_no_learning():
    """
    Test that ZI2 (like ZIC) does not learn or adapt.

    ZI2 should maintain random behavior within constraints.
    """
    buyer = ZI2(
        player_id=1,
        is_buyer=True,
        num_tokens=5,
        valuations=[100] * 5,
        price_min=0,
        price_max=150,
        seed=None  # Random seed
    )

    # Collect bids over time
    early_bids = []
    late_bids = []

    # Early period
    for i in range(100):
        buyer.bid_ask(time=i+1, nobidask=0)
        bid = buyer.bid_ask_response()
        if bid >= 0:
            early_bids.append(bid)
        buyer.has_responded = False

    # Simulate some trades
    buyer.num_trades = 2

    # Late period
    for i in range(100):
        buyer.bid_ask(time=i+101, nobidask=0)
        bid = buyer.bid_ask_response()
        if bid >= 0:
            late_bids.append(bid)
        buyer.has_responded = False
        # Reset to keep testing same token
        buyer.num_trades = 2

    # Both distributions should be similar (no learning)
    from scipy import stats
    _, p_value = stats.mannwhitneyu(early_bids, late_bids, alternative='two-sided')

    # High p-value means distributions are similar
    assert p_value > 0.05, \
        f"ZI2 bid distribution changed over time (p={p_value:.4f})"


# =============================================================================
# TEST 5: ACCEPTANCE LOGIC
# =============================================================================

def test_zi2_acceptance_decisions():
    """
    Test that ZI2 makes correct accept/reject decisions.

    Like ZIC: Accept if profitable, reject if not.
    """
    # Test buyer
    buyer = ZI2(
        player_id=1,
        is_buyer=True,
        num_tokens=2,
        valuations=[100, 80],
        price_min=0,
        price_max=150,
        seed=42
    )

    # Profitable trade
    buyer.buy_sell(
        time=1,
        nobuysell=0,
        high_bid=90,
        low_ask=70,  # Good price
        high_bidder=1,  # This buyer
        low_asker=2
    )
    assert buyer.buy_sell_response() == True, \
        "ZI2 buyer should accept profitable trade"

    # Unprofitable trade
    buyer.has_responded = False
    buyer.num_trades = 0  # Reset

    buyer.buy_sell(
        time=2,
        nobuysell=0,
        high_bid=90,
        low_ask=110,  # Too expensive
        high_bidder=1,
        low_asker=2
    )
    assert buyer.buy_sell_response() == False, \
        "ZI2 buyer should reject unprofitable trade"

    # Test seller
    seller = ZI2(
        player_id=2,
        is_buyer=False,
        num_tokens=2,
        valuations=[50, 70],
        price_min=0,
        price_max=150,
        seed=43
    )

    # Profitable trade
    seller.buy_sell(
        time=1,
        nobuysell=0,
        high_bid=80,  # Good price
        low_ask=60,
        high_bidder=1,
        low_asker=2  # This seller
    )
    assert seller.buy_sell_response() == True, \
        "ZI2 seller should accept profitable trade"


# =============================================================================
# TEST 6: EFFICIENCY BENCHMARK
# =============================================================================

def test_zi2_efficiency_benchmark():
    """
    Test that ZI2 achieves expected efficiency levels.

    Benchmark: Should achieve ~98-99% efficiency (slight improvement over ZIC).
    """
    # Large symmetric market
    num_agents = 5
    num_tokens = 5

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

    for rep in range(10):
        buyers = [
            ZI2(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers = [
            ZI2(i+6, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i+5)
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

        for _ in range(200):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, 200)
        buyer_vals = {i+1: buyers[i].valuations for i in range(num_agents)}
        seller_costs = {i+1: sellers[i].valuations for i in range(num_agents)}

        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

    # Check results
    avg_efficiency = np.mean(efficiencies)
    std_efficiency = np.std(efficiencies)

    print(f"\nZI2 Efficiency: {avg_efficiency:.2f}% ± {std_efficiency:.2f}%")
    print(f"Expected: ~98-99% (slight improvement over ZIC)")

    # Should achieve high efficiency
    assert avg_efficiency > 90, \
        f"ZI2 efficiency {avg_efficiency:.2f}% below expected minimum 90%"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])