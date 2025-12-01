"""
Behavioral tests for Gamer trader.

Verifies the Gamer implementation matches the specification:
- Buyer: floor(0.9 * value) - 10% below value
- Seller: ceil(1.1 * cost) - 10% above cost
- Acceptance: Only when holding the current bid/ask AND spread crossed
- No time pressure, no history, no re-pricing
"""

from traders.legacy.gamer import Gamer


class TestQuoteCalculation:
    """Tests for fixed margin quote calculation."""

    def test_buyer_bids_10_percent_below_value(self):
        """Buyer should bid at floor(0.9 * value)."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        # floor(0.9 * 100) = floor(90) = 90
        assert bid == 90, f"Expected 90, got {bid}"

    def test_buyer_bids_floor_truncation(self):
        """Buyer should use floor (truncation) for non-integer results."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[95, 85, 75, 65],
            price_min=0,
            price_max=200,
        )
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        # floor(0.9 * 95) = floor(85.5) = 85
        assert bid == 85, f"Expected 85, got {bid}"

    def test_seller_asks_10_percent_above_cost_ceil(self):
        """Seller should ask at ceil(1.1 * cost)."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        # ceil(1.1 * 50) = ceil(55.00000000000001) = 56 (floating point)
        assert ask == 56, f"Expected 56, got {ask}"

    def test_seller_asks_ceil_rounds_up(self):
        """Seller should use ceil for non-integer results."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[45, 55, 65, 75],
            price_min=0,
            price_max=200,
        )
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        # ceil(1.1 * 45) = ceil(49.5) = 50
        assert ask == 50, f"Expected 50, got {ask}"

    def test_buyer_price_clamped_to_min(self):
        """Buyer bid should be clamped to price_min."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[5, 10, 15, 20],
            price_min=10,
            price_max=200,
        )
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        # floor(0.9 * 5) = 4, clamped to 10
        assert bid == 10, f"Expected 10 (clamped), got {bid}"

    def test_seller_price_clamped_to_max(self):
        """Seller ask should be clamped to price_max."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[180, 170, 160, 150],
            price_min=0,
            price_max=200,
        )
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        # ceil(1.1 * 180) = ceil(198.00000000000003) = 199 (floating point)
        assert ask == 199, f"Expected 199, got {ask}"

        # Now test with higher cost where clamping kicks in
        seller2 = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[190, 180, 170, 160],
            price_min=0,
            price_max=200,
        )
        seller2.bid_ask(time=1, nobidask=0)
        ask2 = seller2.bid_ask_response()
        # ceil(1.1 * 190) = ceil(209.00...) = 210, clamped to 200
        assert ask2 == 200, f"Expected 200 (clamped), got {ask2}"


class TestNobidaskFlag:
    """Tests for nobidask flag handling."""

    def test_buyer_returns_0_when_nobidask_positive(self):
        """Buyer should return 0 when nobidask > 0."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        buyer.bid_ask(time=1, nobidask=1)
        bid = buyer.bid_ask_response()
        assert bid == 0, f"Expected 0 when nobidask=1, got {bid}"

    def test_seller_returns_0_when_nobidask_positive(self):
        """Seller should return 0 when nobidask > 0."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        seller.bid_ask(time=1, nobidask=1)
        ask = seller.bid_ask_response()
        assert ask == 0, f"Expected 0 when nobidask=1, got {ask}"


class TestMarketImprovement:
    """Tests for market improvement checks."""

    def test_buyer_no_bid_if_cannot_improve(self):
        """Buyer should not bid if target <= current_bid."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        # Set current bid higher than our target (90)
        buyer.current_bid = 95
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        assert bid == 0, f"Expected 0 (cannot improve bid), got {bid}"

    def test_buyer_bids_when_can_improve(self):
        """Buyer should bid if target > current_bid."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        # Set current bid lower than our target (90)
        buyer.current_bid = 85
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        assert bid == 90, f"Expected 90 (can improve), got {bid}"

    def test_seller_no_ask_if_cannot_improve(self):
        """Seller should not ask if target >= current_ask."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        # Set current ask lower than our target (55)
        seller.current_ask = 50
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        assert ask == 0, f"Expected 0 (cannot improve ask), got {ask}"

    def test_seller_asks_when_can_improve(self):
        """Seller should ask if target < current_ask."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        # Set current ask higher than our target (ceil(1.1*50)=56)
        seller.current_ask = 60
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        assert ask == 56, f"Expected 56 (can improve), got {ask}"

    def test_buyer_bids_when_no_current_bid(self):
        """Buyer should bid when current_bid is 0 (no standing bid)."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        buyer.current_bid = 0
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        assert bid == 90, f"Expected 90, got {bid}"

    def test_seller_asks_when_no_current_ask(self):
        """Seller should ask when current_ask is 0 (no standing ask)."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        seller.current_ask = 0
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        # ceil(1.1 * 50) = 56 (floating point)
        assert ask == 56, f"Expected 56, got {ask}"


class TestAcceptance:
    """Tests for trade acceptance logic - CRITICAL."""

    def test_buyer_only_accepts_if_standing_bidder(self):
        """Buyer should only accept if player_id == current_bidder."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        # We are the standing bidder, spread is crossed, profitable
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=90,
            low_ask=85,  # Profitable (value 100 > ask 85)
            high_bidder=1,  # We are the bidder
            low_asker=2,
        )
        accept = buyer.buy_sell_response()
        assert accept is True, "Should accept when standing bidder and profitable"

    def test_buyer_rejects_if_not_bidder_even_if_profitable(self):
        """Buyer should reject even profitable trade if not standing bidder."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        # Someone else is the bidder, even though trade would be profitable
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=90,
            low_ask=85,  # Profitable (value 100 > ask 85)
            high_bidder=2,  # NOT us!
            low_asker=3,
        )
        accept = buyer.buy_sell_response()
        assert accept is False, "Should reject when not the standing bidder"

    def test_seller_only_accepts_if_standing_asker(self):
        """Seller should only accept if player_id == current_asker."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        # We are the standing asker, spread is crossed, profitable
        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=60,  # Profitable (bid 60 > cost 50)
            low_ask=55,
            high_bidder=2,
            low_asker=1,  # We are the asker
        )
        accept = seller.buy_sell_response()
        assert accept is True, "Should accept when standing asker and profitable"

    def test_seller_rejects_if_not_asker_even_if_profitable(self):
        """Seller should reject even profitable trade if not standing asker."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        # Someone else is the asker, even though trade would be profitable
        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=60,  # Profitable (bid 60 > cost 50)
            low_ask=55,
            high_bidder=2,
            low_asker=3,  # NOT us!
        )
        accept = seller.buy_sell_response()
        assert accept is False, "Should reject when not the standing asker"

    def test_buyer_rejects_when_spread_not_crossed(self):
        """Buyer should reject when spread is not crossed (bid < ask)."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        # We are the bidder but spread not crossed
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=80,  # bid < ask
            low_ask=85,
            high_bidder=1,  # We are the bidder
            low_asker=2,
        )
        accept = buyer.buy_sell_response()
        assert accept is False, "Should reject when spread not crossed"

    def test_never_trades_at_loss_buyer(self):
        """Buyer should never accept when ask >= value (loss)."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        # We are the bidder but price is at or above value
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=105,
            low_ask=100,  # ask == value (loss/break-even)
            high_bidder=1,
            low_asker=2,
        )
        accept = buyer.buy_sell_response()
        assert accept is False, "Should reject when ask >= value"

    def test_never_trades_at_loss_seller(self):
        """Seller should never accept when bid <= cost (loss)."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        # We are the asker but price is at or below cost
        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=50,  # bid == cost (loss/break-even)
            low_ask=45,
            high_bidder=2,
            low_asker=1,
        )
        accept = seller.buy_sell_response()
        assert accept is False, "Should reject when bid <= cost"


class TestNobuysellFlag:
    """Tests for nobuysell flag handling."""

    def test_buyer_rejects_when_nobuysell_positive(self):
        """Buyer should reject when nobuysell > 0."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        # nobuysell > 0 means no more tokens available
        buyer.buy_sell(
            time=1,
            nobuysell=1,  # Flag set
            high_bid=90,
            low_ask=85,
            high_bidder=1,  # We are the bidder
            low_asker=2,
        )
        accept = buyer.buy_sell_response()
        assert accept is False, "Should reject when nobuysell > 0"

    def test_seller_rejects_when_nobuysell_positive(self):
        """Seller should reject when nobuysell > 0."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        # nobuysell > 0 means no more tokens available
        seller.buy_sell(
            time=1,
            nobuysell=1,  # Flag set
            high_bid=60,
            low_ask=55,
            high_bidder=2,
            low_asker=1,  # We are the asker
        )
        accept = seller.buy_sell_response()
        assert accept is False, "Should reject when nobuysell > 0"


class TestTokensExhausted:
    """Tests for token exhaustion handling."""

    def test_buyer_no_bid_when_tokens_exhausted(self):
        """Buyer should return 0 when all tokens traded."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=2,
            valuations=[100, 90],
            price_min=0,
            price_max=200,
        )
        buyer.num_trades = 2  # All tokens used
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        assert bid == 0, "Should return 0 when tokens exhausted"

    def test_seller_no_ask_when_tokens_exhausted(self):
        """Seller should return 0 when all tokens traded."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=2,
            valuations=[50, 60],
            price_min=0,
            price_max=200,
        )
        seller.num_trades = 2  # All tokens used
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        assert ask == 0, "Should return 0 when tokens exhausted"

    def test_buyer_rejects_trade_when_tokens_exhausted(self):
        """Buyer should reject trades when all tokens traded."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=2,
            valuations=[100, 90],
            price_min=0,
            price_max=200,
        )
        buyer.num_trades = 2  # All tokens used
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=90,
            low_ask=85,
            high_bidder=1,
            low_asker=2,
        )
        accept = buyer.buy_sell_response()
        assert accept is False, "Should reject when tokens exhausted"


class TestNoStandingQuote:
    """Tests for handling no standing bid/ask."""

    def test_buyer_rejects_when_no_ask(self):
        """Buyer should reject when there is no standing ask."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=90,
            low_ask=0,  # No standing ask
            high_bidder=1,
            low_asker=0,
        )
        accept = buyer.buy_sell_response()
        assert accept is False, "Should reject when no standing ask"

    def test_seller_rejects_when_no_bid(self):
        """Seller should reject when there is no standing bid."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=0,  # No standing bid
            low_ask=55,
            high_bidder=0,
            low_asker=1,
        )
        accept = seller.buy_sell_response()
        assert accept is False, "Should reject when no standing bid"


class TestMarginParameter:
    """Tests for configurable margin parameter."""

    def test_custom_margin_buyer(self):
        """Buyer should use custom margin if specified."""
        buyer = Gamer(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
            margin=0.20,  # 20% margin
        )
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        # floor(0.8 * 100) = 80
        assert bid == 80, f"Expected 80 with 20% margin, got {bid}"

    def test_custom_margin_seller(self):
        """Seller should use custom margin if specified."""
        seller = Gamer(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
            margin=0.20,  # 20% margin
        )
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        # ceil(1.2 * 50) = ceil(60.0) = 60
        assert ask == 60, f"Expected 60 with 20% margin, got {ask}"
