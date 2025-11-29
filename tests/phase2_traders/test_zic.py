"""
Tests for ZIC (Zero Intelligence Constrained) Agent.
"""

import pytest
import numpy as np
from traders.legacy.zic import ZIC

def test_zic_initialization() -> None:
    """Test ZIC initialization."""
    agent = ZIC(1, True, 1, [100], seed=42)
    assert agent.player_id == 1
    assert agent.is_buyer
    assert agent.num_tokens == 1
    assert agent.valuations == [100]
    assert isinstance(agent.rng, np.random.Generator)

def test_zic_bid_range() -> None:
    """Test that ZIC bids are within [min_price, valuation]."""
    # Buyer with valuation 100, min_price 0
    agent = ZIC(1, True, 100, [100] * 100, price_min=0, price_max=200, seed=42)
    
    for _ in range(100):
        bid = agent.bid_ask_response()
        assert 0 <= bid <= 100
        
def test_zic_ask_range() -> None:
    """Test that ZIC asks are within [cost, max_price]."""
    # Seller with cost 50, max_price 100
    agent = ZIC(1, False, 100, [50] * 100, price_min=0, price_max=100, seed=42)
    
    for _ in range(100):
        ask = agent.bid_ask_response()
        assert 50 <= ask <= 100

def test_zic_buy_acceptance() -> None:
    """Test ZIC buy acceptance logic."""
    # Buyer with valuation 100
    agent = ZIC(1, True, 1, [100], seed=42)
    
    # Case 1: Profitable, Winner, Spread Crossed -> Accept
    # Bid 90, Ask 80. Spread crossed.
    agent.buy_sell(1, 0, high_bid=90, low_ask=80, high_bidder=1, low_asker=2)
    assert agent.buy_sell_response() is True
    
    # Case 2: Not Profitable (Valuation <= Ask) -> Reject
    # Ask 100. Profit 0.
    agent.buy_sell(1, 0, high_bid=100, low_ask=100, high_bidder=1, low_asker=2)
    assert agent.buy_sell_response() is False
    
    # Case 3: Not Winner -> Reject
    # Bid 90, Ask 80. But winner is Player 3.
    agent.buy_sell(1, 0, high_bid=90, low_ask=80, high_bidder=3, low_asker=2)
    assert agent.buy_sell_response() is False
    
    # Case 4: Spread Not Crossed -> Reject
    # Bid 70, Ask 80.
    agent.buy_sell(1, 0, high_bid=70, low_ask=80, high_bidder=1, low_asker=2)
    assert agent.buy_sell_response() is False

def test_zic_sell_acceptance() -> None:
    """Test ZIC sell acceptance logic."""
    # Seller with cost 50
    agent = ZIC(1, False, 1, [50], seed=42)
    
    # Case 1: Profitable, Winner, Spread Crossed -> Accept
    # Bid 60, Ask 55. Spread crossed.
    agent.buy_sell(1, 0, high_bid=60, low_ask=55, high_bidder=2, low_asker=1)
    assert agent.buy_sell_response() is True
    
    # Case 2: Not Profitable (Bid <= Cost) -> Reject
    # Bid 50. Profit 0.
    agent.buy_sell(1, 0, high_bid=50, low_ask=50, high_bidder=2, low_asker=1)
    assert agent.buy_sell_response() is False
    
    # Case 3: Not Winner -> Reject
    agent.buy_sell(1, 0, high_bid=60, low_ask=55, high_bidder=2, low_asker=3)
    assert agent.buy_sell_response() is False
    
    # Case 4: Spread Not Crossed -> Reject
    # Bid 40, Ask 55.
    agent.buy_sell(1, 0, high_bid=40, low_ask=55, high_bidder=2, low_asker=1)
    assert agent.buy_sell_response() is False

def test_zic_reproducibility() -> None:
    """Test that ZIC is reproducible with seed."""
    agent1 = ZIC(1, True, 10, [100]*10, seed=42)
    agent2 = ZIC(1, True, 10, [100]*10, seed=42)

    bids1 = [agent1.bid_ask_response() for _ in range(10)]
    bids2 = [agent2.bid_ask_response() for _ in range(10)]

    assert bids1 == bids2

    agent3 = ZIC(1, True, 10, [100]*10, seed=43)
    bids3 = [agent3.bid_ask_response() for _ in range(10)]
    assert bids1 != bids3

def test_zic_boundary_valuation_buyer() -> None:
    """Test buyer with valuation equal to min_price (edge case)."""
    # Buyer with valuation = min_price = 0
    agent = ZIC(1, True, 10, [0] * 10, price_min=0, price_max=100, seed=42)

    # Should bid min_price (0) when valuation == min_price
    for _ in range(10):
        bid = agent.bid_ask_response()
        assert bid == 0

def test_zic_boundary_valuation_seller() -> None:
    """Test seller with cost equal to max_price (edge case)."""
    # Seller with cost = max_price = 100
    agent = ZIC(1, False, 10, [100] * 10, price_min=0, price_max=100, seed=42)

    # Should ask max_price (100) when cost == max_price
    for _ in range(10):
        ask = agent.bid_ask_response()
        assert ask == 100

def test_zic_token_exhaustion() -> None:
    """Test behavior when agent runs out of tokens."""
    # Agent with only 2 tokens
    agent = ZIC(1, True, 2, [100, 80], price_min=0, price_max=100, seed=42)

    # First two responses should be valid
    bid1 = agent.bid_ask_response()
    assert 0 <= bid1 <= 100

    # Simulate a trade (increment num_trades manually)
    agent.num_trades = 1
    bid2 = agent.bid_ask_response()
    assert 0 <= bid2 <= 80

    # After using all tokens, should return 0
    agent.num_trades = 2
    bid3 = agent.bid_ask_response()
    assert bid3 == 0

def test_zic_nobuysell_flag() -> None:
    """Test that agent respects nobuysell flag."""
    agent = ZIC(1, True, 1, [100], seed=42)

    # When nobuysell > 0, should always reject
    agent.buy_sell(1, nobuysell=1, high_bid=90, low_ask=80, high_bidder=1, low_asker=2)
    assert agent.buy_sell_response() is False

    # When nobuysell = 0, normal logic applies
    agent.buy_sell(1, nobuysell=0, high_bid=90, low_ask=80, high_bidder=1, low_asker=2)
    assert agent.buy_sell_response() is True


def test_zic_java_rng_formula_buyer() -> None:
    """
    Test that buyer bid generation matches Java SRobotZI1 formula.

    Java formula: newbid = token - (int)(drand() * (token - minprice))
    This should produce bids in [minprice, token] with a bias toward higher bids.
    """
    # Create buyer with known seed and valuation
    agent = ZIC(1, True, 100, [80] * 100, price_min=20, price_max=100, seed=12345)

    # Generate many bids to check distribution
    bids = []
    for _ in range(1000):
        agent.num_trades = 0  # Reset to use first valuation
        bid = agent.bid_ask_response()
        bids.append(bid)

    # All bids should be in valid range [20, 80]
    assert all(20 <= b <= 80 for b in bids), "All bids should be in [min, valuation]"

    # Check that we get the full range (not all the same value)
    unique_bids = set(bids)
    assert len(unique_bids) > 10, "Should generate diverse bids, not just a few values"

    # Statistical check: Mean should be closer to upper bound due to Java formula
    # Java formula: V - floor(random * (V - min)) biases toward V
    # For V=80, min=20: mean should be around 50-60 (closer to 80 than to 20)
    mean_bid = sum(bids) / len(bids)
    assert 45 <= mean_bid <= 65, f"Mean bid {mean_bid} should be in [45, 65] for Java formula"

    # The bias comes from: V - floor(random * range)
    # When random is small (near 0), bid is near V
    # When random is large (near 1), bid is near min
    # But floor() truncates, creating slight bias toward V


def test_zic_java_rng_formula_seller() -> None:
    """
    Test that seller ask generation matches Java SRobotZI1 formula.

    Java formula: newask = token + (int)(drand() * (maxprice - token))
    This should produce asks in [token, maxprice] with a bias toward lower asks.
    """
    # Create seller with known seed and cost
    agent = ZIC(1, False, 100, [40] * 100, price_min=0, price_max=100, seed=12345)

    # Generate many asks to check distribution
    asks = []
    for _ in range(1000):
        agent.num_trades = 0  # Reset to use first cost
        ask = agent.bid_ask_response()
        asks.append(ask)

    # All asks should be in valid range [40, 100]
    assert all(40 <= a <= 100 for a in asks), "All asks should be in [cost, max]"

    # Check that we get the full range
    unique_asks = set(asks)
    assert len(unique_asks) > 10, "Should generate diverse asks"

    # Statistical check: Mean should be closer to lower bound due to Java formula
    # Java formula: C + floor(random * (max - C)) biases toward C
    # For C=40, max=100: mean should be around 60-75 (closer to 40)
    mean_ask = sum(asks) / len(asks)
    assert 60 <= mean_ask <= 75, f"Mean ask {mean_ask} should be in [60, 75] for Java formula"


def test_zic_formula_boundary_cases() -> None:
    """Test Java formula with boundary valuations."""
    # Buyer with V = min + 1 (very small range)
    agent = ZIC(1, True, 1, [21], price_min=20, price_max=100, seed=42)
    bids = [agent.bid_ask_response() for _ in range(100)]
    # With range=1, formula: 21 - floor(random * 1)
    # random * 1 is in [0, 1), so floor gives either 0
    # Result: mostly 21, sometimes 20
    assert all(b in [20, 21] for b in bids)

    # Seller with C = max - 1 (very small range)
    agent = ZIC(1, False, 1, [99], price_min=0, price_max=100, seed=42)
    asks = [agent.bid_ask_response() for _ in range(100)]
    # With range=1, formula: 99 + floor(random * 1)
    # Result: mostly 99, sometimes 100
    assert all(a in [99, 100] for a in asks)
