"""
Tests for ZIP (Zero Intelligence Plus) Agent.

Validates implementation against Cliff & Bruten 1997 paper.
"""

import pytest
import numpy as np
from traders.legacy.zip import ZIP


def test_zip_initialization() -> None:
    """Test ZIP initialization and parameter ranges."""
    agent = ZIP(1, True, 1, [100], seed=42)

    assert agent.player_id == 1
    assert agent.is_buyer
    assert agent.num_tokens == 1
    assert agent.valuations == [100]

    # Check parameter ranges from paper
    assert 0.1 <= agent.beta <= 0.5  # Learning rate
    assert 0.0 <= agent.gamma <= 0.1  # Momentum

    # Buyer margin should be negative
    assert -0.35 <= agent.margin <= -0.05

    # Check momentum is initialized to zero
    assert agent.momentum_delta == 0.0

    # Verify RNG is set up
    assert isinstance(agent.rng, np.random.Generator)


def test_zip_seller_initialization() -> None:
    """Test ZIP seller initialization."""
    agent = ZIP(2, False, 1, [50], seed=42)

    assert not agent.is_buyer
    # Seller margin should be positive
    assert 0.05 <= agent.margin <= 0.35


def test_zip_shout_price_calculation() -> None:
    """Test shout price calculation: p = λ × (1 + μ)"""
    # Buyer with negative margin
    buyer = ZIP(1, True, 1, [100], price_min=0, price_max=200, seed=42)
    buyer.margin = -0.2

    quote = buyer._calculate_quote()
    expected = 100 * (1.0 - 0.2)  # = 80
    assert quote == 80

    # Seller with positive margin
    seller = ZIP(2, False, 1, [50], price_min=0, price_max=200, seed=42)
    seller.margin = 0.3

    quote = seller._calculate_quote()
    expected = 50 * (1.0 + 0.3)  # = 65
    assert quote == 65


def test_zip_shout_price_clamping() -> None:
    """Test that shout prices are clamped to [price_min, price_max]."""
    # Buyer with high negative margin (would go below min)
    buyer = ZIP(1, True, 1, [10], price_min=0, price_max=100, seed=42)
    buyer.margin = -0.9  # 10 * (1 - 0.9) = 1

    quote = buyer._calculate_quote()
    assert quote >= 0  # Should be clamped

    # Seller with high positive margin (would go above max)
    seller = ZIP(2, False, 1, [90], price_min=0, price_max=100, seed=42)
    seller.margin = 5.0  # 90 * (1 + 5.0) = 540

    quote = seller._calculate_quote()
    assert quote <= 100  # Should be clamped


def test_zip_target_price_raise_margin() -> None:
    """Test target price calculation when raising margin."""
    agent = ZIP(1, True, 1, [100], seed=42)

    # When raising margin: R ~ [1.0, 1.05], A ~ [0.0, 0.05]
    # So target should be slightly higher than last_price
    last_price = 50
    target = agent._calculate_target_price(last_price, raise_margin=True)

    # Target should be > last_price and ≤ last_price * 1.05 + 0.05
    assert target >= last_price
    assert target <= last_price * 1.05 + 0.05


def test_zip_target_price_lower_margin() -> None:
    """Test target price calculation when lowering margin."""
    agent = ZIP(1, True, 1, [100], seed=42)

    # When lowering margin: R ~ [0.95, 1.0], A ~ [-0.05, 0.0]
    # So target should be slightly lower than last_price
    last_price = 50
    target = agent._calculate_target_price(last_price, raise_margin=False)

    # Target should be ≥ last_price * 0.95 - 0.05 and ≤ last_price
    assert target >= last_price * 0.95 - 0.05
    assert target <= last_price


def test_zip_target_price_clamping() -> None:
    """Test that target prices are clamped to valid range."""
    agent = ZIP(1, True, 1, [100], price_min=10, price_max=90, seed=42)

    # Last price very high, raise margin
    target = agent._calculate_target_price(1000, raise_margin=True)
    assert target <= agent.price_max

    # Last price very low, lower margin
    target = agent._calculate_target_price(1, raise_margin=False)
    assert target >= agent.price_min


def test_zip_margin_update_math() -> None:
    """Test margin update equations (13, 15, 16) from paper."""
    agent = ZIP(1, True, 1, [100], seed=42)

    # Set known parameters
    agent.beta = 0.3
    agent.gamma = 0.05
    agent.margin = -0.2
    agent.momentum_delta = 0.0

    # Current price: 100 * (1 - 0.2) = 80
    # Target price: 85 (simulated)
    target_price = 85.0

    current_price = agent._calculate_quote()  # Should be 80
    assert current_price == 80

    # Apply update
    agent._update_margin(target_price)

    # Verify math:
    # delta = 0.3 * (85 - 80) = 1.5
    # momentum_delta = 0.05 * 0 + 0.95 * 1.5 = 1.425
    # new_price = 80 + 1.425 = 81.425
    # new_margin = (81.425 / 100) - 1 = -0.18575

    expected_delta = 0.3 * (85 - 80)  # = 1.5
    expected_momentum = 0.05 * 0.0 + 0.95 * expected_delta  # = 1.425
    expected_new_price = 80 + expected_momentum  # = 81.425
    expected_margin = (expected_new_price / 100.0) - 1.0  # = -0.18575

    assert abs(agent.momentum_delta - expected_momentum) < 0.001
    assert abs(agent.margin - expected_margin) < 0.001


def test_zip_margin_clamping_buyer() -> None:
    """Test that buyer margins are clamped to [-0.99, 0.0]."""
    buyer = ZIP(1, True, 1, [100], seed=42)

    # Force margin to go positive (invalid for buyers)
    buyer.margin = 0.5
    buyer._update_margin(150.0)  # Very high target

    # Should be clamped to 0.0
    assert buyer.margin <= 0.0
    assert buyer.margin >= -0.99


def test_zip_margin_clamping_seller() -> None:
    """Test that seller margins are clamped to [0.0, 10.0]."""
    seller = ZIP(2, False, 1, [50], seed=42)

    # Force margin to go negative (invalid for sellers)
    seller.margin = -0.1
    seller._update_margin(10.0)  # Very low target

    # Should be clamped to 0.0
    assert seller.margin >= 0.0
    assert seller.margin <= 10.0


def test_zip_acceptance_buyer_shout_price() -> None:
    """Test buyer acceptance uses shout price, not limit price."""
    buyer = ZIP(1, True, 1, [100], price_min=0, price_max=200, seed=42)
    buyer.margin = -0.2

    # Calculate shout price: 100 * (1 - 0.2) = 80
    buyer.current_quote = buyer._calculate_quote()
    assert buyer.current_quote == 80

    # Simulate bid/ask stage
    buyer.bid_ask(1, 0)
    buyer.current_quote = 80  # Ensure quote is set

    # Simulate buy/sell stage
    # Buyer is high bidder with bid=80, low ask=75
    buyer.buy_sell(1, 0, high_bid=80, low_ask=75, high_bidder=1, low_asker=2)

    # Per paper Section 4.1: Accept if offer (75) ≤ my bid shout price (80)
    result = buyer.buy_sell_response()
    assert result is True  # Should accept

    # Case 2: Low ask > my shout price
    buyer.buy_sell(1, 0, high_bid=80, low_ask=85, high_bidder=1, low_asker=2)
    result = buyer.buy_sell_response()
    assert result is False  # Should reject


def test_zip_acceptance_seller_shout_price() -> None:
    """Test seller acceptance uses shout price, not limit price."""
    seller = ZIP(2, False, 1, [50], price_min=0, price_max=200, seed=42)
    seller.margin = 0.3

    # Calculate shout price: 50 * (1 + 0.3) = 65
    seller.current_quote = seller._calculate_quote()
    assert seller.current_quote == 65

    # Simulate bid/ask stage
    seller.bid_ask(1, 0)
    seller.current_quote = 65  # Ensure quote is set

    # Simulate buy/sell stage
    # Seller is low asker with ask=65, high bid=70
    seller.buy_sell(1, 0, high_bid=70, low_ask=65, high_bidder=1, low_asker=2)

    # Per paper Section 4.1: Accept if bid (70) ≥ my ask shout price (65)
    result = seller.buy_sell_response()
    assert result is True  # Should accept

    # Case 2: High bid < my shout price
    seller.buy_sell(1, 0, high_bid=60, low_ask=65, high_bidder=1, low_asker=2)
    result = seller.buy_sell_response()
    assert result is False  # Should reject


def test_zip_acceptance_not_winner() -> None:
    """Test that agent rejects if not the winner."""
    buyer = ZIP(1, True, 1, [100], seed=42)
    buyer.current_quote = 80

    buyer.bid_ask(1, 0)

    # Buyer is NOT high bidder (player 3 is)
    buyer.buy_sell(1, 0, high_bid=85, low_ask=75, high_bidder=3, low_asker=2)

    result = buyer.buy_sell_response()
    assert result is False  # Should reject


def test_zip_acceptance_spread_not_crossed() -> None:
    """Test that agent rejects if spread not crossed."""
    buyer = ZIP(1, True, 1, [100], seed=42)
    buyer.current_quote = 80

    buyer.bid_ask(1, 0)

    # Spread not crossed: bid=80, ask=90
    buyer.buy_sell(1, 0, high_bid=80, low_ask=90, high_bidder=1, low_asker=2)

    result = buyer.buy_sell_response()
    assert result is False  # Should reject


def test_zip_token_exhaustion() -> None:
    """Test behavior when agent runs out of tokens."""
    agent = ZIP(1, True, 2, [100, 80], seed=42)

    # First quote should work
    quote1 = agent._calculate_quote()
    assert quote1 > 0

    # Simulate a trade
    agent.num_trades = 1
    quote2 = agent._calculate_quote()
    assert quote2 > 0

    # After using all tokens
    agent.num_trades = 2
    quote3 = agent._calculate_quote()
    assert quote3 == 0


def test_zip_reproducibility() -> None:
    """Test that ZIP is reproducible with seed."""
    agent1 = ZIP(1, True, 5, [100]*5, seed=42)
    agent2 = ZIP(1, True, 5, [100]*5, seed=42)

    # Should have same parameters
    assert agent1.beta == agent2.beta
    assert agent1.gamma == agent2.gamma
    assert agent1.margin == agent2.margin

    # Should generate same quotes
    quotes1 = [agent1._calculate_quote() for _ in range(5)]
    quotes2 = [agent2._calculate_quote() for _ in range(5)]
    assert quotes1 == quotes2

    # Different seed should give different results
    agent3 = ZIP(1, True, 5, [100]*5, seed=99)
    assert agent3.beta != agent1.beta or agent3.gamma != agent1.gamma


def test_zip_should_raise_margin_buyer() -> None:
    """Test buyer raise margin logic per Figure 27."""
    buyer = ZIP(1, True, 1, [100], seed=42)

    # Scenario: Trade accepted, my bid was >= trade price
    buyer.last_shout_accepted = True
    my_price = 90
    trade_price = 85

    result = buyer._should_raise_margin(my_price, trade_price)
    # Buyer: if my_price >= trade_price, should raise (I bid too high)
    assert result is True

    # Scenario: My bid was < trade price
    result = buyer._should_raise_margin(80, 85)
    assert result is False


def test_zip_should_raise_margin_seller() -> None:
    """Test seller raise margin logic per Figure 27."""
    seller = ZIP(2, False, 1, [50], seed=42)

    # Scenario: Trade accepted, my ask was <= trade price
    seller.last_shout_accepted = True
    my_price = 60
    trade_price = 70

    result = seller._should_raise_margin(my_price, trade_price)
    # Seller: if my_price <= trade_price, should raise (I asked too low)
    assert result is True

    # Scenario: My ask was > trade price
    result = seller._should_raise_margin(75, 70)
    assert result is False


def test_zip_should_lower_margin_buyer() -> None:
    """Test buyer lower margin logic per Figure 27."""
    buyer = ZIP(1, True, 1, [100], seed=42)
    buyer.num_trades = 0  # Active

    # Scenario: Trade occurred, last shout was ask, my price <= last price
    buyer.last_shout_accepted = True
    buyer.last_shout_was_bid = False  # Last shout was ask
    my_price = 70
    last_price = 80

    result = buyer._should_lower_margin(my_price, last_price)
    # Buyer: if active and last was ask and my_price <= last_price, lower
    assert result is True

    # Scenario: My price > last price
    result = buyer._should_lower_margin(85, 80)
    assert result is False


def test_zip_should_lower_margin_seller() -> None:
    """Test seller lower margin logic per Figure 27."""
    seller = ZIP(2, False, 1, [50], seed=42)
    seller.num_trades = 0  # Active

    # Scenario: Trade occurred, last shout was bid, my price >= last price
    seller.last_shout_accepted = True
    seller.last_shout_was_bid = True  # Last shout was bid
    my_price = 80
    last_price = 70

    result = seller._should_lower_margin(my_price, last_price)
    # Seller: if active and last was bid and my_price >= last_price, lower
    assert result is True

    # Scenario: My price < last price
    result = seller._should_lower_margin(65, 70)
    assert result is False


def test_zip_no_update_when_inactive() -> None:
    """Test that inactive agents (out of tokens) don't lower margin."""
    buyer = ZIP(1, True, 2, [100, 90], seed=42)
    buyer.num_trades = 2  # No more tokens

    buyer.last_shout_accepted = True
    buyer.last_shout_was_bid = False

    result = buyer._should_lower_margin(70, 80)
    assert result is False  # Inactive agents don't lower
