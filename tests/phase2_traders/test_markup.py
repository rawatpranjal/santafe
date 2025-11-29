"""
Tests for Markup Agent - Fixed percentage markup strategy.
"""

import pytest
from traders.legacy.markup import Markup


def test_markup_initialization() -> None:
    """Test Markup initialization."""
    agent = Markup(1, True, 1, [100])
    assert agent.player_id == 1
    assert agent.is_buyer
    assert agent.num_tokens == 1
    assert agent.valuations == [100]
    assert agent.markup_pct == 0.10  # Default 10%


def test_markup_custom_percentage() -> None:
    """Test Markup with custom percentage."""
    agent = Markup(1, True, 1, [100], markup_pct=0.20)
    assert agent.markup_pct == 0.20


def test_markup_buyer_bid() -> None:
    """Test that buyer bids at valuation * (1 - markup%)."""
    # Buyer with valuation 100, 10% markup -> bid 90
    agent = Markup(1, True, 1, [100], markup_pct=0.10)
    bid = agent.bid_ask_response()
    assert bid == 90  # 100 * (1 - 0.10) = 90

    # Buyer with valuation 80, 25% markup -> bid 60
    agent = Markup(1, True, 1, [80], markup_pct=0.25)
    bid = agent.bid_ask_response()
    assert bid == 60  # 80 * (1 - 0.25) = 60


def test_markup_seller_ask() -> None:
    """Test that seller asks at cost * (1 + markup%)."""
    # Seller with cost 50, 10% markup -> ask 55
    agent = Markup(1, False, 1, [50], markup_pct=0.10)
    ask = agent.bid_ask_response()
    assert ask == 55  # 50 * (1 + 0.10) = 55

    # Seller with cost 40, 25% markup -> ask 50
    agent = Markup(1, False, 1, [40], markup_pct=0.25)
    ask = agent.bid_ask_response()
    assert ask == 50  # 40 * (1 + 0.25) = 50


def test_markup_deterministic() -> None:
    """Test that Markup is deterministic (no randomness)."""
    agent1 = Markup(1, True, 3, [100, 80, 60])
    agent2 = Markup(2, True, 3, [100, 80, 60])

    # Both should produce identical bids
    for _ in range(3):
        bid1 = agent1.bid_ask_response()
        bid2 = agent2.bid_ask_response()
        assert bid1 == bid2


def test_markup_buy_acceptance() -> None:
    """Test Markup buy acceptance logic."""
    # Buyer with valuation 100
    agent = Markup(1, True, 1, [100])

    # Case 1: Profitable, Winner, Spread Crossed -> Accept
    agent.buy_sell(1, 0, high_bid=90, low_ask=80, high_bidder=1, low_asker=2)
    assert agent.buy_sell_response() is True

    # Case 2: Not Profitable (Valuation <= Ask) -> Reject
    agent.buy_sell(1, 0, high_bid=100, low_ask=100, high_bidder=1, low_asker=2)
    assert agent.buy_sell_response() is False

    # Case 3: Not Winner -> Reject
    agent.buy_sell(1, 0, high_bid=90, low_ask=80, high_bidder=3, low_asker=2)
    assert agent.buy_sell_response() is False


def test_markup_sell_acceptance() -> None:
    """Test Markup sell acceptance logic."""
    # Seller with cost 50
    agent = Markup(1, False, 1, [50])

    # Case 1: Profitable, Winner, Spread Crossed -> Accept
    agent.buy_sell(1, 0, high_bid=60, low_ask=55, high_bidder=2, low_asker=1)
    assert agent.buy_sell_response() is True

    # Case 2: Not Profitable (Bid <= Cost) -> Reject
    agent.buy_sell(1, 0, high_bid=50, low_ask=50, high_bidder=2, low_asker=1)
    assert agent.buy_sell_response() is False

    # Case 3: Not Winner -> Reject
    agent.buy_sell(1, 0, high_bid=60, low_ask=55, high_bidder=2, low_asker=3)
    assert agent.buy_sell_response() is False


def test_markup_price_bounds() -> None:
    """Test that bids/asks respect price bounds."""
    # Buyer: bid should be at least min_price
    agent = Markup(1, True, 1, [5], price_min=10, markup_pct=0.50)
    bid = agent.bid_ask_response()
    assert bid >= 10  # Clamped to min_price

    # Seller: ask should be at most max_price
    agent = Markup(1, False, 1, [95], price_max=100, markup_pct=0.50)
    ask = agent.bid_ask_response()
    assert ask <= 100  # Clamped to max_price


def test_markup_token_exhaustion() -> None:
    """Test behavior when agent runs out of tokens."""
    agent = Markup(1, True, 2, [100, 80])

    # First bid should work
    bid1 = agent.bid_ask_response()
    assert bid1 == 90  # 100 * 0.9

    # Simulate trade
    agent.num_trades = 1
    bid2 = agent.bid_ask_response()
    assert bid2 == 72  # 80 * 0.9

    # After using all tokens, should return 0
    agent.num_trades = 2
    bid3 = agent.bid_ask_response()
    assert bid3 == 0
