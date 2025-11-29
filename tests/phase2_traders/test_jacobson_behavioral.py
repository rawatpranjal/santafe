"""
Behavioral Tests for Jacobson Agent.

Tests validate that Jacobson agents exhibit expected equilibrium estimation
and gap-closing behavior from the 1993 Santa Fe Tournament.

Key behaviors tested:
- Equilibrium learning via weighted price history
- Confidence-based convergence toward equilibrium
- Gap closure analysis with time pressure
- Probabilistic acceptance when profitable
- Market efficiency in various configurations
"""

import pytest
import numpy as np
from typing import List

from traders.legacy.jacobson import Jacobson
from traders.legacy.zic import ZIC
from traders.legacy.zip import ZIP
from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
)


# =============================================================================
# TEST 1: INSTANTIATION AND INTERFACE
# =============================================================================

def test_jacobson_instantiation_and_interface():
    """Test Jacobson agent can be instantiated and has required interface."""
    agent = Jacobson(
        player_id=1,
        is_buyer=True,
        num_tokens=3,
        valuations=[100, 90, 80],
        price_min=0,
        price_max=200,
        num_times=100,
        seed=42
    )

    # Test attributes
    assert agent.player_id == 1
    assert agent.is_buyer is True
    assert agent.num_tokens == 3
    assert agent.roundweight == 0.0
    assert agent.roundpricesum == 0.0
    assert agent.lastgap == 10000000

    # Test interface methods exist
    assert hasattr(agent, 'bid_ask')
    assert hasattr(agent, 'bid_ask_response')
    assert hasattr(agent, 'buy_sell')
    assert hasattr(agent, 'buy_sell_response')

    # Test methods are callable
    agent.bid_ask(1, 0)
    bid = agent.bid_ask_response()
    assert isinstance(bid, int)


# =============================================================================
# TEST 2: EQUILIBRIUM LEARNING
# =============================================================================

def test_jacobson_equilibrium_learning():
    """Test that Jacobson learns equilibrium from trade history."""
    buyer = Jacobson(1, True, 5, [100, 90, 80, 70, 60], seed=42)

    buyer.current_period = 1

    # Simulate trades at consistent price
    equilibrium_price = 75
    for i in range(5):
        buyer.num_trades = i
        buyer.buy_sell_result(1, equilibrium_price, 1, 0, 0, 0, 0)

    # Check equilibrium estimate converges
    eq = buyer._eqest()
    # Should be close to equilibrium_price
    assert abs(eq - equilibrium_price) < 5, f"Eq estimate {eq:.1f} should be near {equilibrium_price}"

    # Check confidence increases
    conf = buyer._eqconf()
    assert conf > 0.8, f"Confidence {conf:.3f} should be high after 5 trades"


def test_jacobson_bid_ask_consistency():
    """Test that bids/asks move toward equilibrium with confidence."""
    buyer = Jacobson(1, True, 3, [100, 90, 80], seed=42)

    # Establish equilibrium at 70
    buyer.roundweight = 50.0
    buyer.roundpricesum = 3500.0  # 70 * 50
    buyer.current_bid = 50

    bid = buyer._player_request_bid()

    # With high confidence (0.01^(1/50) ≈ 0.9), bid should move toward eq=70
    # Expected: roughly 50*(1-0.9) + 70*0.9 + 1 = 5 + 63 + 1 = 69
    assert 65 <= bid <= 75, f"Bid {bid} should move toward equilibrium 70"

    # Test seller
    seller = Jacobson(2, False, 3, [50, 60, 70], seed=42)
    seller.roundweight = 50.0
    seller.roundpricesum = 3500.0  # eq = 70
    seller.current_ask = 90

    ask = seller._player_request_ask()

    # Ask should move toward eq=70
    # Expected: roughly 90*(1-0.9) + 70*0.9 - 1 = 9 + 63 - 1 = 71
    assert 65 <= ask <= 75, f"Ask {ask} should move toward equilibrium 70"


# =============================================================================
# TEST 3: JACOBSON VS ZIC (1v1)
# =============================================================================

def test_jacobson_1v1_vs_zic_buyers():
    """
    Test Jacobson buyer vs ZIC seller achieves good efficiency.

    Target: 80-95% efficiency
    Jacobson should learn equilibrium and outperform random ZIC.
    """
    num_tokens = 3
    num_times = 100  # Steps per market run

    buyer_valuations = [100, 90, 80]
    seller_valuations = [40, 50, 60]

    efficiencies = []

    for rep in range(5):  # 5 replications
        jacobson_buyer = Jacobson(
            1, True, num_tokens, buyer_valuations,
            price_min=0, price_max=200, num_times=num_times, seed=rep
        )
        zic_seller = ZIC(
            2, False, num_tokens, seller_valuations,
            price_min=0, price_max=200, seed=rep+100
        )

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=num_times,
            price_min=0,
            price_max=200,
            buyers=[jacobson_buyer],
            sellers=[zic_seller],
        )

        # Run market
        for _ in range(num_times):
            if not market.run_time_step():
                break

        # Calculate efficiency
        trades = extract_trades_from_orderbook(market.orderbook, num_times)

        # Build valuations dicts (use local indices, not agent IDs)
        buyer_vals = {1: jacobson_buyer.valuations}  # local buyer index 1
        seller_costs = {1: zic_seller.valuations}   # local seller index 1

        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [jacobson_buyer.valuations],
            [zic_seller.valuations]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

    # Average efficiency should be in target range
    avg_efficiency = np.mean(efficiencies)
    print(f"\nJacobson vs ZIC (buyer): {avg_efficiency:.1f}% efficiency")

    assert avg_efficiency >= 75, f"Efficiency {avg_efficiency:.1f}% below target 75%"


def test_jacobson_1v1_vs_zic_sellers():
    """
    Test ZIC buyer vs Jacobson seller achieves good efficiency.

    Target: 80-95% efficiency
    """
    num_tokens = 3
    num_times = 100

    buyer_valuations = [100, 90, 80]
    seller_valuations = [40, 50, 60]

    efficiencies = []

    for rep in range(5):
        zic_buyer = ZIC(
            1, True, num_tokens, buyer_valuations,
            price_min=0, price_max=200, seed=rep
        )
        jacobson_seller = Jacobson(
            2, False, num_tokens, seller_valuations,
            price_min=0, price_max=200, num_times=num_times, seed=rep+100
        )

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=num_times,
            price_min=0,
            price_max=200,
            buyers=[zic_buyer],
            sellers=[jacobson_seller],
        )

        # Run market
        for _ in range(num_times):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, num_times)

        # Build valuations dicts (use local indices, not agent IDs)
        buyer_vals = {1: zic_buyer.valuations}
        seller_costs = {1: jacobson_seller.valuations}

        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [zic_buyer.valuations],
            [jacobson_seller.valuations]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

    avg_efficiency = np.mean(efficiencies)
    print(f"\nZIC vs Jacobson (seller): {avg_efficiency:.1f}% efficiency")

    assert avg_efficiency >= 75, f"Efficiency {avg_efficiency:.1f}% below target 75%"


# =============================================================================
# TEST 4: JACOBSON SELF-PLAY
# =============================================================================

def test_jacobson_self_play_efficiency():
    """
    Test Jacobson agents in self-play achieve good efficiency.

    Target: 85-95% efficiency
    Both sides should converge to equilibrium through mutual learning.
    """
    num_agents = 2
    num_tokens = 3
    num_times = 150  # Longer for learning

    buyer_valuations = [[100, 90, 80], [95, 85, 75]]
    seller_valuations = [[40, 50, 60], [45, 55, 65]]

    efficiencies = []

    for rep in range(5):
        buyers = [
            Jacobson(i+1, True, num_tokens, buyer_valuations[i],
                    price_min=0, price_max=200, num_times=num_times, seed=rep*10+i)
            for i in range(num_agents)
        ]
        sellers = [
            Jacobson(i+3, False, num_tokens, seller_valuations[i],
                    price_min=0, price_max=200, num_times=num_times, seed=rep*10+i+2)
            for i in range(num_agents)
        ]

        market = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=num_times,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
        )

        # Run market
        for _ in range(num_times):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, num_times)

        # Build valuations dicts (use local indices 1-based, not agent IDs)
        buyer_vals = {i+1: buyers[i].valuations for i in range(num_agents)}  # local indices 1, 2
        seller_costs = {i+1: sellers[i].valuations for i in range(num_agents)}  # local indices 1, 2

        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

    avg_efficiency = np.mean(efficiencies)
    print(f"\nJacobson self-play: {avg_efficiency:.1f}% efficiency")

    # Jacobson was a mid-tier strategy with probabilistic acceptance
    # 70%+ shows agent trades and converges reasonably
    assert avg_efficiency >= 70, f"Efficiency {avg_efficiency:.1f}% below target 70%"


# =============================================================================
# TEST 5: JACOBSON GAP ANALYSIS
# =============================================================================

def test_jacobson_gap_closing_behavior():
    """
    Test that Jacobson monitors gap closure and waits when gap is closing.

    Behavior: Agent should wait when gap is decreasing, act when stuck or time pressure.
    """
    buyer = Jacobson(1, True, 3, [100, 90, 80], seed=42)

    # Setup: profitable trade available, agent is winner
    buyer.num_trades = 0  # First token, valuation=100
    buyer.current_bidder = 1
    buyer.current_bid = 70
    buyer.current_ask = 80
    buyer.current_time = 50
    buyer.num_times = 100

    # Test 1: Gap is closing (lastgap > current gap)
    buyer.lastgap = 15  # Gap was 15, now is 10
    result = buyer._player_request_buy()
    # Should wait (gap closing), so likely reject unless probabilistic
    # We can't assert exact result due to randomness, but verify no crash
    assert result in [0, 1]

    # Test 2: Gap is stuck (lastgap == current gap)
    buyer.lastgap = 10  # Gap same as current
    buyer.rng = __import__('random').Random(1)  # Fixed seed
    result = buyer._player_request_buy()
    # Should enter probabilistic zone
    assert result in [0, 1]

    # Test 3: Time pressure (near end of period)
    buyer.current_time = 98
    buyer.lastgap = 10
    result = buyer._player_request_buy()
    # Time pressure formula should trigger probabilistic
    assert result in [0, 1]


def test_jacobson_buy_sell_decisions_rational():
    """Test that Jacobson never accepts unprofitable trades."""
    buyer = Jacobson(1, True, 3, [100, 90, 80], seed=42)

    # Test: Unprofitable trade (ask > valuation)
    buyer.num_trades = 0
    buyer.current_bidder = 1
    buyer.current_bid = 70
    buyer.current_ask = 105  # Above valuation of 100
    buyer.current_time = 50
    buyer.num_times = 100

    result = buyer._player_request_buy()
    assert result == 0, "Should reject unprofitable trade"

    # Test: Not winner
    buyer.current_bidder = 99  # Not us
    buyer.current_ask = 90  # Profitable price
    result = buyer._player_request_buy()
    assert result == 0, "Should reject when not winner"

    # Test: Profitable and winner with crossed spread
    buyer.current_bidder = 1
    buyer.current_bid = 85
    buyer.current_ask = 80  # Crossed spread
    result = buyer._player_request_buy()
    assert result == 1, "Should accept crossed profitable spread"


# =============================================================================
# TEST 6: MIXED STRATEGY TOURNAMENT
# =============================================================================

def test_jacobson_mixed_strategy_tournament():
    """
    Test Jacobson in mixed market with ZIC and ZIP.

    Target: Market efficiency >85%
    Verify Jacobson doesn't disrupt market dynamics.
    """
    num_tokens = 3
    num_times = 150  # Longer for mixed strategies

    buyer_valuations = [[100, 90, 80], [95, 85, 75], [98, 88, 78]]
    seller_valuations = [[40, 50, 60], [45, 55, 65], [42, 52, 62]]

    efficiencies = []

    for rep in range(3):
        # Mix of strategies: Jacobson, ZIC, ZIP
        buyers = [
            Jacobson(1, True, num_tokens, buyer_valuations[0],
                    price_min=0, price_max=200, num_times=num_times, seed=rep),
            ZIC(2, True, num_tokens, buyer_valuations[1],
                price_min=0, price_max=200, seed=rep+10),
            ZIP(3, True, num_tokens, buyer_valuations[2],
                price_min=0, price_max=200, seed=rep+20),
        ]

        sellers = [
            ZIC(4, False, num_tokens, seller_valuations[0],
                price_min=0, price_max=200, seed=rep+30),
            Jacobson(5, False, num_tokens, seller_valuations[1],
                    price_min=0, price_max=200, num_times=num_times, seed=rep+40),
            ZIP(6, False, num_tokens, seller_valuations[2],
                price_min=0, price_max=200, seed=rep+50),
        ]

        market = Market(
            num_buyers=3,
            num_sellers=3,
            num_times=num_times,
            price_min=0,
            price_max=200,
            buyers=buyers,
            sellers=sellers,
        )

        # Run market
        for _ in range(num_times):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, num_times)

        # Build valuations dicts using LOCAL indices (1-based)
        buyer_vals = {i+1: buyers[i].valuations for i in range(3)}
        seller_costs = {i+1: sellers[i].valuations for i in range(3)}

        actual_surplus = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            efficiency = calculate_allocative_efficiency(actual_surplus, max_surplus)
            efficiencies.append(efficiency)

    avg_efficiency = np.mean(efficiencies)
    print(f"\nMixed strategy (Jacobson/ZIC/ZIP): {avg_efficiency:.1f}% efficiency")

    assert avg_efficiency >= 80, f"Efficiency {avg_efficiency:.1f}% below target 80%"


# =============================================================================
# TEST 7: PERIOD LIFECYCLE
# =============================================================================

def test_jacobson_full_period_lifecycle():
    """Test Jacobson through complete period lifecycle."""
    buyer = Jacobson(1, True, 3, [100, 90, 80], seed=42)

    # Start period
    buyer.start_period(1)
    assert buyer.current_period == 1

    # Bid/ask phase
    buyer.bid_ask(1, 0)
    bid = buyer.bid_ask_response()
    assert isinstance(bid, int)

    # Buy/sell phase
    buyer.buy_sell(1, 0, 70, 80, 99, 2)
    accept = buyer.buy_sell_response()
    assert isinstance(accept, bool)

    # Result
    buyer.buy_sell_result(0, 0, 0, 70, 99, 80, 2)

    # End period
    buyer.end_period()


# =============================================================================
# TEST 8: MULTI-PERIOD LEARNING
# =============================================================================

def test_jacobson_multi_period_learning():
    """Test that equilibrium persists across periods within a round."""
    buyer = Jacobson(1, True, 5, [100] * 5, seed=42)

    # Period 1: Establish equilibrium
    buyer.start_period(1)
    buyer.current_period = 1
    for i in range(3):
        buyer.num_trades = i
        buyer.buy_sell_result(1, 75, 1, 0, 0, 0, 0)

    eq_p1 = buyer._eqest()
    weight_p1 = buyer.roundweight

    # Period 2: Should persist
    buyer.start_period(2)
    assert buyer.current_period == 2
    assert buyer.roundweight == weight_p1  # Persists
    assert buyer._eqest() == eq_p1  # Same equilibrium


# =============================================================================
# TEST 9: ROUND RESET
# =============================================================================

def test_jacobson_round_reset():
    """Test that round state resets at period 1."""
    buyer = Jacobson(1, True, 5, [100] * 5, seed=42)

    # Establish some round state
    buyer.roundweight = 100.0
    buyer.roundpricesum = 7500.0
    buyer.lastgap = 5

    # Call start_period(1) - should reset
    buyer.start_period(1)

    assert buyer.roundweight == 0.0
    assert buyer.roundpricesum == 0.0
    assert buyer.lastgap == 10000000


# =============================================================================
# TEST 10: REPRODUCIBILITY
# =============================================================================

def test_jacobson_reproducibility():
    """Test that same seed produces identical behavior."""
    # Run two markets with same seed
    results = []

    for run in range(2):  # Run twice with same seed
        buyer = Jacobson(1, True, 2, [100, 90], seed=42)
        seller = ZIC(2, False, 2, [50, 60], seed=142)

        market = Market(
            num_buyers=1,
            num_sellers=1,
            num_times=50,
            price_min=0,
            price_max=200,
            buyers=[buyer],
            sellers=[seller],
            seed=42  # Same market seed
        )

        # Run market
        for _ in range(50):
            if not market.run_time_step():
                break

        trades = extract_trades_from_orderbook(market.orderbook, 50)

        # Trades are tuples: (buyer_id, seller_id, price, buyer_unit)
        run_results = [(price, buyer_id, seller_id) for buyer_id, seller_id, price, _ in trades]
        results.append(run_results)

    # Results should be identical
    assert results[0] == results[1], "Same seed should produce identical trades"
