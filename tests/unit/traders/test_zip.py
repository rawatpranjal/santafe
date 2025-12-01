# tests/unit/traders/test_zip.py
"""
Adversarial tests for ZIP (Zero-Intelligence Plus) agent.

ZIP from Cliff & Bruten (1997) uses adaptive learning via Widrow-Hoff rule.
These tests verify:
1. Quote formula: p = λ × (1 + μ)
2. Margin constraints (sellers ≥ 0, buyers ≤ 0)
3. Learning direction (raise/lower based on market signals)
4. Budget constraints (never trade at a loss)
"""


from traders.legacy.zip import ZIP

# =============================================================================
# Test: Quote Calculation (p = λ × (1 + μ))
# =============================================================================


class TestQuoteCalculation:
    """Tests for the ZIP quote formula."""

    def test_seller_positive_margin_increases_price(self):
        """Seller's positive margin should result in quote > limit price."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
            margin=0.10,  # 10% markup
        )

        # Force margin to known value
        seller.margin = 0.10

        seller.bid_ask(time=1, nobidask=0)
        quote = seller.bid_ask_response()

        # Quote should be 50 * 1.10 = 55
        expected = int(round(50 * 1.10))
        assert quote == expected, f"Expected quote {expected}, got {quote}"

    def test_buyer_negative_margin_decreases_price(self):
        """Buyer's negative margin should result in quote < limit price."""
        buyer = ZIP(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
            margin=-0.10,  # -10% markdown
        )

        # Force margin to known value
        buyer.margin = -0.10

        buyer.bid_ask(time=1, nobidask=0)
        quote = buyer.bid_ask_response()

        # Quote should be 100 * 0.90 = 90
        expected = int(round(100 * 0.90))
        assert quote == expected, f"Expected quote {expected}, got {quote}"

    def test_zero_margin_quote_equals_limit(self):
        """With zero margin, quote should equal limit price."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=1,
            valuations=[75],
            price_min=1,
            price_max=200,
            seed=42,
        )

        seller.margin = 0.0
        seller.bid_ask(time=1, nobidask=0)
        quote = seller.bid_ask_response()

        assert quote == 75, f"Expected quote 75 with zero margin, got {quote}"

    def test_quote_clamped_to_price_range(self):
        """Quote should be clamped to [min_price, max_price]."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=1,
            valuations=[180],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # High margin would push quote above max_price
        seller.margin = 0.50  # 180 * 1.5 = 270 > 200
        seller.bid_ask(time=1, nobidask=0)
        quote = seller.bid_ask_response()

        assert quote <= 200, f"Quote {quote} exceeds max_price 200"


# =============================================================================
# Test: Margin Constraints
# =============================================================================


class TestMarginConstraints:
    """Tests for margin constraint enforcement."""

    def test_seller_margin_stays_non_negative(self):
        """Seller margin must never go below 0."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Simulate many margin updates to try to force negative
        for _ in range(100):
            # Use extreme target that would push margin negative
            seller._update_margin(target_price=10.0)  # Very low target

        assert seller.margin >= 0.0, f"Seller margin went negative: {seller.margin}"

    def test_buyer_margin_stays_non_positive(self):
        """Buyer margin must never go above 0."""
        buyer = ZIP(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Force initial negative margin
        buyer.margin = -0.01

        # Simulate many margin updates to try to force positive
        for _ in range(100):
            # Use extreme target that would push margin positive
            buyer._update_margin(target_price=200.0)  # Very high target

        assert buyer.margin <= 0.0, f"Buyer margin went positive: {buyer.margin}"

    def test_margin_clamped_to_reasonable_range(self):
        """Margin should be clamped to prevent extreme values."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        seller.margin = 10.0  # Unreasonably high
        seller._update_margin(target_price=100.0)

        # After update, margin should be clamped
        assert seller.margin <= 0.50, f"Margin {seller.margin} exceeds reasonable bound"


# =============================================================================
# Test: Learning Direction
# =============================================================================


class TestLearningDirection:
    """Tests for correct learning direction based on market signals."""

    def test_should_raise_margin_after_accepted_trade(self):
        """After trade, should raise margin if could have gotten better price."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        seller.last_shout_accepted = True
        # My ask was 60, trade at 70 - I could have asked higher
        my_price = 60
        trade_price = 70

        result = seller._should_raise_margin(my_price, trade_price)
        assert result is True, "Seller should raise margin when sold below trade price"

    def test_should_not_raise_margin_when_trade_rejected(self):
        """Should not raise margin if no trade occurred."""
        buyer = ZIP(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        buyer.last_shout_accepted = False  # No trade

        result = buyer._should_raise_margin(80, 75)
        assert result is False, "Should not raise margin when no trade occurred"


# =============================================================================
# Test: Profitable Trade Only
# =============================================================================


class TestProfitableTradesOnly:
    """Tests that ZIP never trades at a loss."""

    def test_buyer_rejects_ask_above_limit(self):
        """Buyer should reject if ask price exceeds limit price."""
        buyer = ZIP(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Set margin that makes quote 95, but ask is above our LIMIT of 100
        buyer.margin = -0.05  # Quote would be 95
        buyer.current_quote = int(100 * 0.95)  # 95

        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=95,
            low_ask=105,  # Above limit price of 100!
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Buyer should reject ask above limit price"

    def test_seller_rejects_bid_below_cost(self):
        """Seller should reject if bid price is below cost."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Quote is 55, but bid is below our COST of 50
        seller.margin = 0.10  # Quote would be 55
        seller.current_quote = int(50 * 1.10)  # 55

        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=45,  # Below cost of 50!
            low_ask=55,
            high_bidder=2,
            low_asker=1,
        )

        result = seller.buy_sell_response()
        assert result is False, "Seller should reject bid below cost"


# =============================================================================
# Test: Trade Acceptance Rules
# =============================================================================


class TestTradeAcceptanceRules:
    """Tests for ZIP's trade acceptance logic (shout price based)."""

    def test_buyer_accepts_when_ask_below_quote(self):
        """Buyer accepts if ask ≤ current_quote AND below limit."""
        buyer = ZIP(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        buyer.margin = -0.10  # Quote = 90
        buyer.current_quote = int(100 * 0.90)

        # Spread crossed, we're high bidder, ask at our quote
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=95,
            low_ask=90,  # Equals quote
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is True, "Buyer should accept when ask ≤ quote"

    def test_buyer_rejects_when_ask_above_quote(self):
        """Buyer rejects if ask > current_quote."""
        buyer = ZIP(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        buyer.margin = -0.10  # Quote = 90
        buyer.current_quote = int(100 * 0.90)

        # Ask above our quote
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=95,
            low_ask=92,  # Above quote of 90
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Buyer should reject when ask > quote"

    def test_seller_accepts_when_bid_above_quote(self):
        """Seller accepts if bid ≥ current_quote AND above cost."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        seller.margin = 0.10  # Quote = 55
        seller.current_quote = int(50 * 1.10)

        # Spread crossed, we're low asker, bid at our quote
        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=55,  # Equals quote
            low_ask=55,
            high_bidder=2,
            low_asker=1,
        )

        result = seller.buy_sell_response()
        assert result is True, "Seller should accept when bid ≥ quote"


# =============================================================================
# Test: start_round() Resets Learning
# =============================================================================


class TestStartRoundReset:
    """Tests that start_round() properly resets learning state."""

    def test_start_round_resets_margin(self):
        """start_round() should reset margin to initial range."""
        agent = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Learn extreme margin
        agent.margin = 0.45  # High

        # Start new round
        agent.start_round([55, 65, 75, 85])

        # Margin should be reset to small value
        assert agent.margin < 0.01, f"Margin {agent.margin} not reset after start_round()"

    def test_start_round_resets_momentum(self):
        """start_round() should reset momentum delta."""
        agent = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Accumulate momentum
        agent.momentum_delta = 5.0

        agent.start_round([55, 65, 75, 85])

        assert agent.momentum_delta == 0.0, "Momentum should be reset"

    def test_start_round_clears_market_state(self):
        """start_round() should clear stale market observations."""
        agent = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        # Set market state
        agent.last_shout_price = 75
        agent.current_high_bid = 80

        agent.start_round([55, 65, 75, 85])

        assert agent.last_shout_price == 0, "last_shout_price should be cleared"
        assert agent.current_high_bid == 0, "current_high_bid should be cleared"


# =============================================================================
# Test: Hyperparameter Overrides
# =============================================================================


class TestHyperparameterOverrides:
    """Tests that hyperparameters can be overridden via kwargs."""

    def test_beta_override(self):
        """beta should be settable via kwargs."""
        agent = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
            beta=0.5,  # Override default
        )

        assert agent.beta == 0.5, f"beta should be 0.5, got {agent.beta}"

    def test_gamma_override(self):
        """gamma should be settable via kwargs."""
        agent = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
            gamma=0.05,
        )

        assert agent.gamma == 0.05, f"gamma should be 0.05, got {agent.gamma}"

    def test_margin_init_override(self):
        """Initial margin should be settable."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
            margin_init=0.20,
        )

        assert seller.margin == 0.20, f"margin should be 0.20, got {seller.margin}"


# =============================================================================
# Test: Spread Must Be Crossed
# =============================================================================


class TestSpreadCrossing:
    """Tests that ZIP only trades when spread is crossed."""

    def test_rejects_when_spread_not_crossed(self):
        """Should reject trade when bid < ask."""
        buyer = ZIP(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        buyer.margin = -0.10
        buyer.current_quote = 90

        # Spread NOT crossed: bid < ask
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=85,
            low_ask=90,  # bid < ask
            high_bidder=1,
            low_asker=2,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Should reject when spread not crossed"

    def test_accepts_when_spread_exactly_crossed(self):
        """Should accept when bid == ask."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        seller.margin = 0.0  # Quote = cost = 50
        seller.current_quote = 50

        # Spread exactly crossed: bid == ask
        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=50,
            low_ask=50,  # bid == ask
            high_bidder=2,
            low_asker=1,
        )

        result = seller.buy_sell_response()
        assert result is True, "Should accept when spread exactly crossed"


# =============================================================================
# Test: No Tokens Left
# =============================================================================


class TestNoTokensLeft:
    """Tests behavior when all tokens have been traded."""

    def test_quote_is_zero_when_no_tokens(self):
        """Quote should be 0 when no tokens left."""
        agent = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=2,
            valuations=[50, 60],
            price_min=1,
            price_max=200,
            seed=42,
        )

        agent.num_trades = 2  # All tokens traded

        agent.bid_ask(time=1, nobidask=0)
        quote = agent.bid_ask_response()

        assert quote == 0, f"Quote should be 0 when no tokens left, got {quote}"

    def test_rejects_trade_when_no_tokens(self):
        """Should reject trade when no tokens left."""
        agent = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=2,
            valuations=[50, 60],
            price_min=1,
            price_max=200,
            seed=42,
        )

        agent.num_trades = 2

        agent.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=70,
            low_ask=60,
            high_bidder=2,
            low_asker=1,
        )

        result = agent.buy_sell_response()
        assert result is False, "Should reject trade when no tokens left"


# =============================================================================
# Test: Must Be Winner To Accept
# =============================================================================


class TestMustBeWinner:
    """Tests that only high bidder / low asker can accept."""

    def test_buyer_rejects_when_not_high_bidder(self):
        """Buyer should reject if not the high bidder."""
        buyer = ZIP(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=1,
            price_max=200,
            seed=42,
        )

        buyer.margin = -0.10
        buyer.current_quote = 90

        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=95,
            low_ask=85,
            high_bidder=2,  # Someone else is high bidder!
            low_asker=3,
        )

        result = buyer.buy_sell_response()
        assert result is False, "Buyer should reject when not high bidder"

    def test_seller_rejects_when_not_low_asker(self):
        """Seller should reject if not the low asker."""
        seller = ZIP(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=1,
            price_max=200,
            seed=42,
        )

        seller.margin = 0.10
        seller.current_quote = 55

        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=60,
            low_ask=55,
            high_bidder=2,
            low_asker=3,  # Someone else is low asker!
        )

        result = seller.buy_sell_response()
        assert result is False, "Seller should reject when not low asker"
