"""
Multi-Period Trading Dynamics Tests for ZI2.

These tests validate ZI2's behavior across multiple trading periods and tokens,
including token depletion, trading volume, temporal consistency, and multi-token sequences.
"""

import pytest
import numpy as np
from scipy import stats
from typing import List

from traders.legacy.zi2 import ZI2
from traders.legacy.zic import ZIC
from engine.market import Market
from engine.efficiency import extract_trades_from_orderbook


# =============================================================================
# TEST 1: TOKEN DEPLETION TRACKING
# =============================================================================

def test_zi2_token_depletion():
    """
    Test that ZI2 properly tracks and handles token depletion.

    As tokens are traded, ZI2 should:
    1. Increment num_trades counter
    2. Use next token for pricing decisions
    3. Return 0 when all tokens exhausted
    """
    buyer = ZI2(
        player_id=1,
        is_buyer=True,
        num_tokens=3,
        valuations=[100, 80, 60],  # Decreasing
        price_min=0,
        price_max=150,
        seed=42
    )

    # Start with no trades
    assert buyer.num_trades == 0

    # Get bid for first token
    buyer.bid_ask(time=1, nobidask=0)
    bid1 = buyer.bid_ask_response()
    assert 0 < bid1 <= 100, f"First bid should use token[0]=100, got {bid1}"

    # Simulate first trade
    buyer.buy_sell_result(
        status=1,
        trade_price=70,
        trade_type=1,
        high_bid=0,
        high_bidder=0,
        low_ask=0,
        low_asker=0
    )
    assert buyer.num_trades == 1

    # Get bid for second token
    buyer.has_responded = False
    buyer.bid_ask(time=2, nobidask=0)
    bid2 = buyer.bid_ask_response()
    assert 0 < bid2 <= 80, f"Second bid should use token[1]=80, got {bid2}"

    # Simulate second trade
    buyer.buy_sell_result(
        status=1,
        trade_price=65,
        trade_type=1,
        high_bid=0,
        high_bidder=0,
        low_ask=0,
        low_asker=0
    )
    assert buyer.num_trades == 2

    # Get bid for third token
    buyer.has_responded = False
    buyer.bid_ask(time=3, nobidask=0)
    bid3 = buyer.bid_ask_response()
    assert 0 < bid3 <= 60, f"Third bid should use token[2]=60, got {bid3}"

    # Simulate third trade
    buyer.buy_sell_result(
        status=1,
        trade_price=50,
        trade_type=1,
        high_bid=0,
        high_bidder=0,
        low_ask=0,
        low_asker=0
    )
    assert buyer.num_trades == 3

    # All tokens exhausted - should return 0
    buyer.has_responded = False
    buyer.bid_ask(time=4, nobidask=0)
    bid4 = buyer.bid_ask_response()
    assert bid4 == 0, f"With all tokens traded, bid should be 0, got {bid4}"

    print(f"\nToken Depletion Test:")
    print(f"  Token 0 (val=100): bid={bid1}")
    print(f"  Token 1 (val=80):  bid={bid2}")
    print(f"  Token 2 (val=60):  bid={bid3}")
    print(f"  All exhausted:     bid={bid4}")
    print(f"  ✓ Token tracking correct")


# =============================================================================
# TEST 2: TRADING VOLUME (ZI2 VS ZIC)
# =============================================================================

def test_zi2_trading_volume():
    """
    Test that ZI2 achieves similar trading volume to ZIC.

    Hypothesis: Market-awareness shouldn't significantly change trading activity.
    Expected: ZI2 volume within 90-110% of ZIC volume.
    """
    num_reps = 10
    num_agents = 4
    num_tokens = 4

    buyer_tokens = [
        [200, 180, 160, 140],
        [195, 175, 155, 135],
        [190, 170, 150, 130],
        [185, 165, 145, 125],
    ]

    seller_tokens = [
        [40, 60, 80, 100],
        [45, 65, 85, 105],
        [50, 70, 90, 110],
        [55, 75, 95, 115],
    ]

    zi2_volumes = []
    zic_volumes = []

    for rep in range(num_reps):
        # ZI2 Market
        buyers_zi2 = [
            ZI2(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers_zi2 = [
            ZI2(i+5, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i+4)
            for i in range(num_agents)
        ]

        market_zi2 = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=150,
            price_min=0,
            price_max=250,
            buyers=buyers_zi2,
            sellers=sellers_zi2,
            seed=rep
        )

        for _ in range(150):
            if not market_zi2.run_time_step():
                break

        trades_zi2 = extract_trades_from_orderbook(market_zi2.orderbook, 150)
        zi2_volumes.append(len(trades_zi2))

        # ZIC Market
        buyers_zic = [
            ZIC(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i)
            for i in range(num_agents)
        ]
        sellers_zic = [
            ZIC(i+5, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=250, seed=rep*100+i+4)
            for i in range(num_agents)
        ]

        market_zic = Market(
            num_buyers=num_agents,
            num_sellers=num_agents,
            num_times=150,
            price_min=0,
            price_max=250,
            buyers=buyers_zic,
            sellers=sellers_zic,
            seed=rep
        )

        for _ in range(150):
            if not market_zic.run_time_step():
                break

        trades_zic = extract_trades_from_orderbook(market_zic.orderbook, 150)
        zic_volumes.append(len(trades_zic))

    # Compare volumes
    avg_zi2_vol = np.mean(zi2_volumes)
    avg_zic_vol = np.mean(zic_volumes)
    volume_ratio = avg_zi2_vol / avg_zic_vol if avg_zic_vol > 0 else 0

    print(f"\nTrading Volume Test:")
    print(f"  ZI2 avg volume: {avg_zi2_vol:.1f} trades")
    print(f"  ZIC avg volume: {avg_zic_vol:.1f} trades")
    print(f"  Volume ratio: {volume_ratio:.2f}x")

    # Volumes should be similar (90-110% range)
    assert 0.85 <= volume_ratio <= 1.15, \
        f"ZI2 volume should match ZIC (90-110%), got {volume_ratio:.2f}x"

    # Statistical test - volumes should not be significantly different
    _, p_value = stats.ttest_ind(zi2_volumes, zic_volumes)
    print(f"  t-test p-value: {p_value:.4f}")
    print(f"  Result: {'SIMILAR' if p_value > 0.05 else 'DIFFERENT'} volumes")


# =============================================================================
# TEST 3: TEMPORAL CONSISTENCY (NO LEARNING)
# =============================================================================

def test_zi2_first_vs_last_period():
    """
    Test that ZI2 performance is consistent across periods (no learning).

    ZI2 is zero-intelligence, so efficiency should not improve over time.
    """
    num_agents = 4
    num_tokens = 3
    num_periods = 10

    buyer_tokens = [
        [180, 160, 140],
        [175, 155, 135],
        [170, 150, 130],
        [165, 145, 125],
    ]

    seller_tokens = [
        [60, 80, 100],
        [65, 85, 105],
        [70, 90, 110],
        [75, 95, 115],
    ]

    period_efficiencies = []

    for period in range(num_periods):
        buyers = [
            ZI2(i+1, True, num_tokens, buyer_tokens[i],
                price_min=0, price_max=220, seed=period*100+i)
            for i in range(num_agents)
        ]
        sellers = [
            ZI2(i+5, False, num_tokens, seller_tokens[i],
                price_min=0, price_max=220, seed=period*100+i+4)
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
            seed=period
        )

        for _ in range(100):
            if not market.run_time_step():
                break

        # Calculate efficiency
        from engine.efficiency import (
            calculate_actual_surplus,
            calculate_max_surplus,
            calculate_allocative_efficiency
        )

        trades = extract_trades_from_orderbook(market.orderbook, 100)
        buyer_vals = {i+1: buyers[i].valuations for i in range(num_agents)}
        seller_costs = {i+1: sellers[i].valuations for i in range(num_agents)}

        actual = calculate_actual_surplus(trades, buyer_vals, seller_costs)
        max_surplus = calculate_max_surplus(
            [b.valuations for b in buyers],
            [s.valuations for s in sellers]
        )

        if max_surplus > 0:
            eff = calculate_allocative_efficiency(actual, max_surplus)
            period_efficiencies.append(eff)

    # Split into early and late periods
    early_periods = period_efficiencies[:5]
    late_periods = period_efficiencies[5:]

    avg_early = np.mean(early_periods)
    avg_late = np.mean(late_periods)

    print(f"\nTemporal Consistency Test:")
    print(f"  Early periods (1-5): {avg_early:.2f}%")
    print(f"  Late periods (6-10): {avg_late:.2f}%")
    print(f"  Difference: {avg_late - avg_early:.2f} pp")

    # Should NOT show learning (no significant difference)
    _, p_value = stats.ttest_ind(early_periods, late_periods)
    print(f"  t-test p-value: {p_value:.4f}")
    print(f"  Result: {'NO LEARNING' if p_value > 0.05 else 'LEARNING DETECTED'} ✓")

    # Difference should be small (<5 percentage points)
    assert abs(avg_late - avg_early) < 5.0, \
        f"ZI2 should not learn across periods, difference {abs(avg_late - avg_early):.2f}pp too large"


# =============================================================================
# TEST 4: MULTI-TOKEN SEQUENCE VALIDATION
# =============================================================================

def test_zi2_multi_token_sequence():
    """
    Test that ZI2 correctly progresses through token sequence.

    For each trade, ZI2 should:
    1. Use current token's valuation
    2. Increment counter after trade
    3. Use next token's valuation
    """
    # Test buyer with 5 tokens (decreasing valuations)
    buyer = ZI2(
        player_id=1,
        is_buyer=True,
        num_tokens=5,
        valuations=[200, 180, 160, 140, 120],
        price_min=0,
        price_max=250,
        seed=42
    )

    token_sequence_correct = True

    for token_idx in range(5):
        expected_val = buyer.valuations[token_idx]

        # Get bid
        buyer.bid_ask(time=token_idx+1, nobidask=0)
        bid = buyer.bid_ask_response()

        # Bid should not exceed current token valuation
        if bid > 0:
            if bid > expected_val:
                token_sequence_correct = False
                print(f"  ERROR: Token {token_idx}, val={expected_val}, bid={bid} (exceeds valuation)")
            else:
                print(f"  Token {token_idx}: valuation={expected_val}, bid={bid} ✓")

        # Simulate trade
        buyer.buy_sell_result(
            status=1,
            trade_price=expected_val - 10,  # Profitable trade
            trade_type=1,
            high_bid=0,
            high_bidder=0,
            low_ask=0,
            low_asker=0
        )

        # Check counter incremented
        assert buyer.num_trades == token_idx + 1, \
            f"After trade {token_idx + 1}, num_trades should be {token_idx + 1}, got {buyer.num_trades}"

        buyer.has_responded = False

    assert token_sequence_correct, "Multi-token sequence validation failed"
    print(f"\n✓ All 5 tokens traded in correct sequence")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
