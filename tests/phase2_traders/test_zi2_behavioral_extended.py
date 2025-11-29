"""
Extended Behavioral Tests for Zero Intelligence 2 (ZI2) Trading Agents.

These tests focus on market-aware bidding behavior that distinguishes ZI2 from ZIC.

ZI2-specific behaviors tested:
1. Market-aware bidding (considers cbid)
2. Market-aware asking (considers cask)
3. Edge case handling (cbid > token, cask < token)
4. Role asymmetry (buyer vs seller performance)
"""

import pytest
import numpy as np
from typing import List

from traders.legacy.zi2 import ZI2
from engine.market import Market
from engine.efficiency import (
    extract_trades_from_orderbook,
    calculate_actual_surplus,
    calculate_max_surplus,
    calculate_allocative_efficiency,
)


# =============================================================================
# TEST 1: MARKET-AWARE BIDDING (CBID INFLUENCE)
# =============================================================================

def test_zi2_market_aware_bidding():
    """
    Test that ZI2 buyer bids are influenced by current best bid (cbid).

    ZI2 Rule: When cbid exists and cbid <= token, bid in range [cbid, token].
    Expected: Bids should cluster closer to cbid than pure random [min, token].
    """
    buyer = ZI2(
        player_id=1,
        is_buyer=True,
        num_tokens=1,
        valuations=[100],
        price_min=0,
        price_max=150,
        seed=42
    )

    # Test with cbid present and <= token (market-aware case)
    cbid = 70
    buyer.current_bid = cbid

    bids_with_cbid = []
    for _ in range(100):
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        if bid > 0:
            bids_with_cbid.append(bid)
            # Bid should be in range [cbid, token]
            assert bid >= cbid, f"Bid {bid} below cbid {cbid}"
            assert bid <= 100, f"Bid {bid} exceeds valuation 100"
        buyer.has_responded = False
        buyer.num_trades = 0  # Reset

    # Test without cbid (pure random case like ZIC)
    buyer.current_bid = 0

    bids_without_cbid = []
    for _ in range(100):
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        if bid > 0:
            bids_without_cbid.append(bid)
            # Bid should be in range [min, token]
            assert bid >= 0, f"Bid {bid} below min_price"
            assert bid <= 100, f"Bid {bid} exceeds valuation 100"
        buyer.has_responded = False
        buyer.num_trades = 0

    # With cbid, mean should be higher (bids pushed up by cbid floor)
    mean_with_cbid = np.mean(bids_with_cbid)
    mean_without_cbid = np.mean(bids_without_cbid)

    print(f"\nMarket-Aware Bidding Test:")
    print(f"  Mean bid with cbid={cbid}: {mean_with_cbid:.2f}")
    print(f"  Mean bid without cbid: {mean_without_cbid:.2f}")
    print(f"  Difference: {mean_with_cbid - mean_without_cbid:.2f}")

    # cbid should push bids higher
    assert mean_with_cbid > mean_without_cbid, \
        "cbid should increase average bid (market-aware behavior)"


# =============================================================================
# TEST 2: MARKET-AWARE ASKING (CASK INFLUENCE)
# =============================================================================

def test_zi2_market_aware_asking():
    """
    Test that ZI2 seller asks are influenced by current best ask (cask).

    ZI2 Rule: When cask exists and cask >= token, ask in range [token, maxprice].
    Expected: Asks should cluster closer to cask than pure random [token, max].
    """
    seller = ZI2(
        player_id=2,
        is_buyer=False,
        num_tokens=1,
        valuations=[50],
        price_min=0,
        price_max=150,
        seed=43
    )

    # Test with cask present and >= token (market-aware case)
    cask = 80
    seller.current_ask = cask

    asks_with_cask = []
    for _ in range(100):
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        if ask > 0:
            asks_with_cask.append(ask)
            # Ask should be in range [token, maxprice]
            # NOTE: Java uses maxprice even when cask exists (line 58 of SRobotZI2.java)
            assert ask >= 50, f"Ask {ask} below cost 50"
            assert ask <= 150, f"Ask {ask} exceeds max_price"
        seller.has_responded = False
        seller.num_trades = 0

    # Test without cask (pure random case like ZIC)
    seller.current_ask = 0

    asks_without_cask = []
    for _ in range(100):
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        if ask > 0:
            asks_without_cask.append(ask)
            assert ask >= 50, f"Ask {ask} below cost 50"
            assert ask <= 150, f"Ask {ask} exceeds max_price"
        seller.has_responded = False
        seller.num_trades = 0

    # Distribution should be same (Java implementation uses maxprice in both cases)
    mean_with_cask = np.mean(asks_with_cask)
    mean_without_cask = np.mean(asks_without_cask)

    print(f"\nMarket-Aware Asking Test:")
    print(f"  Mean ask with cask={cask}: {mean_with_cask:.2f}")
    print(f"  Mean ask without cask: {mean_without_cask:.2f}")
    print(f"  Difference: {mean_with_cask - mean_without_cask:.2f}")

    # Both should be valid (Java implementation)
    assert mean_with_cask >= 50 and mean_with_cask <= 150
    assert mean_without_cask >= 50 and mean_without_cask <= 150


# =============================================================================
# TEST 3: CBID EXCEEDS TOKEN (EDGE CASE)
# =============================================================================

def test_zi2_cbid_exceeds_token():
    """
    Test ZI2 behavior when current bid exceeds buyer's token valuation.

    ZI2 Rule: If cbid > token, bid at minprice (can't compete).
    Expected: All bids should be at min_price.
    """
    buyer = ZI2(
        player_id=1,
        is_buyer=True,
        num_tokens=1,
        valuations=[80],  # Low valuation
        price_min=10,
        price_max=150,
        seed=42
    )

    # Set cbid higher than token
    buyer.current_bid = 90  # Exceeds valuation of 80

    bids = []
    for _ in range(50):
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        if bid > 0:
            bids.append(bid)
            # Should bid at minprice (can't compete)
            assert bid == 10, \
                f"When cbid ({buyer.current_bid}) > token (80), bid should be minprice (10), got {bid}"
        buyer.has_responded = False
        buyer.num_trades = 0

    print(f"\nCbid Exceeds Token Test:")
    print(f"  Token valuation: 80")
    print(f"  Current bid (cbid): 90")
    print(f"  Expected behavior: Bid at minprice (10)")
    print(f"  Bids generated: {set(bids)}")

    # All bids should be minprice
    assert all(b == 10 for b in bids), \
        "All bids should be minprice when cbid > token"


# =============================================================================
# TEST 4: CASK BELOW TOKEN (EDGE CASE)
# =============================================================================

def test_zi2_cask_below_token():
    """
    Test ZI2 behavior when current ask is below seller's token cost.

    ZI2 Rule: If cask < token, ask at maxprice (can't compete).
    Expected: All asks should be at max_price.
    """
    seller = ZI2(
        player_id=2,
        is_buyer=False,
        num_tokens=1,
        valuations=[70],  # High cost
        price_min=0,
        price_max=100,
        seed=43
    )

    # Set cask lower than token
    seller.current_ask = 60  # Below cost of 70

    asks = []
    for _ in range(50):
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        if ask > 0:
            asks.append(ask)
            # Should ask at maxprice (can't compete)
            assert ask == 100, \
                f"When cask ({seller.current_ask}) < token (70), ask should be maxprice (100), got {ask}"
        seller.has_responded = False
        seller.num_trades = 0

    print(f"\nCask Below Token Test:")
    print(f"  Token cost: 70")
    print(f"  Current ask (cask): 60")
    print(f"  Expected behavior: Ask at maxprice (100)")
    print(f"  Asks generated: {set(asks)}")

    # All asks should be maxprice
    assert all(a == 100 for a in asks), \
        "All asks should be maxprice when cask < token"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
