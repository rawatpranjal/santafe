"""
Behavioral tests for TruthTeller trader.

Verifies the TruthTeller implementation matches the specification:
- Buyer: bids exactly at token value
- Seller: asks exactly at token cost
- Accepts only STRICTLY profitable trades (no breakeven)
- Must be holding the current bid/ask to accept
- No learning, no prediction, no randomization
"""

from traders.legacy.truth_teller import TruthTeller


class TestQuoteCalculation:
    """Tests for truthful quote calculation."""

    def test_buyer_bids_exact_value(self):
        """Buyer should bid at exactly token value."""
        buyer = TruthTeller(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        assert bid == 100, f"Expected 100 (exact value), got {bid}"

    def test_seller_asks_exact_cost(self):
        """Seller should ask at exactly token cost."""
        seller = TruthTeller(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        assert ask == 50, f"Expected 50 (exact cost), got {ask}"

    def test_buyer_price_clamped_to_max(self):
        """Buyer bid should be clamped to price_max."""
        buyer = TruthTeller(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[250, 200, 150, 100],
            price_min=0,
            price_max=200,
        )
        buyer.bid_ask(time=1, nobidask=0)
        bid = buyer.bid_ask_response()
        assert bid == 200, f"Expected 200 (clamped to max), got {bid}"

    def test_seller_price_clamped_to_min(self):
        """Seller ask should be clamped to price_min."""
        seller = TruthTeller(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[5, 10, 15, 20],
            price_min=10,
            price_max=200,
        )
        seller.bid_ask(time=1, nobidask=0)
        ask = seller.bid_ask_response()
        assert ask == 10, f"Expected 10 (clamped to min), got {ask}"

    def test_quotes_for_subsequent_tokens(self):
        """Should quote correct value for each token in sequence."""
        buyer = TruthTeller(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        # First token
        buyer.bid_ask(time=1, nobidask=0)
        assert buyer.bid_ask_response() == 100

        # After trading first token
        buyer.num_trades = 1
        buyer.bid_ask(time=2, nobidask=0)
        assert buyer.bid_ask_response() == 90

        # After trading second token
        buyer.num_trades = 2
        buyer.bid_ask(time=3, nobidask=0)
        assert buyer.bid_ask_response() == 80


class TestNobidaskFlag:
    """Tests for nobidask flag handling."""

    def test_buyer_returns_0_when_nobidask_positive(self):
        """Buyer should return 0 when nobidask > 0."""
        buyer = TruthTeller(
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
        seller = TruthTeller(
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


class TestStrictProfitability:
    """Tests for strictly profitable trades only (no breakeven)."""

    def test_buyer_rejects_breakeven_trade(self):
        """Buyer should reject when ask == value (breakeven)."""
        buyer = TruthTeller(
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
            high_bid=100,
            low_ask=100,  # ask == value (breakeven)
            high_bidder=1,
            low_asker=2,
        )
        accept = buyer.buy_sell_response()
        assert accept is False, "Should reject breakeven trade (ask == value)"

    def test_seller_rejects_breakeven_trade(self):
        """Seller should reject when bid == cost (breakeven)."""
        seller = TruthTeller(
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
            high_bid=50,  # bid == cost (breakeven)
            low_ask=50,
            high_bidder=2,
            low_asker=1,
        )
        accept = seller.buy_sell_response()
        assert accept is False, "Should reject breakeven trade (bid == cost)"

    def test_buyer_accepts_strictly_profitable(self):
        """Buyer should accept when ask < value (strictly profitable)."""
        buyer = TruthTeller(
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
            high_bid=95,
            low_ask=95,  # ask < value (strictly profitable)
            high_bidder=1,
            low_asker=2,
        )
        accept = buyer.buy_sell_response()
        assert accept is True, "Should accept strictly profitable trade (ask < value)"

    def test_seller_accepts_strictly_profitable(self):
        """Seller should accept when bid > cost (strictly profitable)."""
        seller = TruthTeller(
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
            high_bid=55,  # bid > cost (strictly profitable)
            low_ask=55,
            high_bidder=2,
            low_asker=1,
        )
        accept = seller.buy_sell_response()
        assert accept is True, "Should accept strictly profitable trade (bid > cost)"


class TestAcceptance:
    """Tests for trade acceptance logic."""

    def test_buyer_only_accepts_if_standing_bidder(self):
        """Buyer should only accept if player_id == current_bidder."""
        buyer = TruthTeller(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        # We are the standing bidder
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=90,
            low_ask=90,
            high_bidder=1,  # We are the bidder
            low_asker=2,
        )
        accept = buyer.buy_sell_response()
        assert accept is True, "Should accept when standing bidder and profitable"

    def test_buyer_rejects_if_not_bidder(self):
        """Buyer should reject even profitable trade if not standing bidder."""
        buyer = TruthTeller(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        # Someone else is the bidder
        buyer.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=90,
            low_ask=90,
            high_bidder=2,  # NOT us!
            low_asker=3,
        )
        accept = buyer.buy_sell_response()
        assert accept is False, "Should reject when not the standing bidder"

    def test_seller_only_accepts_if_standing_asker(self):
        """Seller should only accept if player_id == current_asker."""
        seller = TruthTeller(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        # We are the standing asker
        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=55,
            low_ask=55,
            high_bidder=2,
            low_asker=1,  # We are the asker
        )
        accept = seller.buy_sell_response()
        assert accept is True, "Should accept when standing asker and profitable"

    def test_seller_rejects_if_not_asker(self):
        """Seller should reject even profitable trade if not standing asker."""
        seller = TruthTeller(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        # Someone else is the asker
        seller.buy_sell(
            time=1,
            nobuysell=0,
            high_bid=60,
            low_ask=55,
            high_bidder=2,
            low_asker=3,  # NOT us!
        )
        accept = seller.buy_sell_response()
        assert accept is False, "Should reject when not the standing asker"

    def test_buyer_rejects_when_spread_not_crossed(self):
        """Buyer should reject when spread is not crossed (bid < ask)."""
        buyer = TruthTeller(
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
            high_bid=80,  # bid < ask (not crossed)
            low_ask=90,
            high_bidder=1,
            low_asker=2,
        )
        accept = buyer.buy_sell_response()
        assert accept is False, "Should reject when spread not crossed"


class TestNobuysellFlag:
    """Tests for nobuysell flag handling."""

    def test_buyer_rejects_when_nobuysell_positive(self):
        """Buyer should reject when nobuysell > 0."""
        buyer = TruthTeller(
            player_id=1,
            is_buyer=True,
            num_tokens=4,
            valuations=[100, 90, 80, 70],
            price_min=0,
            price_max=200,
        )
        buyer.buy_sell(
            time=1,
            nobuysell=1,  # Flag set
            high_bid=90,
            low_ask=90,
            high_bidder=1,
            low_asker=2,
        )
        accept = buyer.buy_sell_response()
        assert accept is False, "Should reject when nobuysell > 0"

    def test_seller_rejects_when_nobuysell_positive(self):
        """Seller should reject when nobuysell > 0."""
        seller = TruthTeller(
            player_id=1,
            is_buyer=False,
            num_tokens=4,
            valuations=[50, 60, 70, 80],
            price_min=0,
            price_max=200,
        )
        seller.buy_sell(
            time=1,
            nobuysell=1,  # Flag set
            high_bid=60,
            low_ask=55,
            high_bidder=2,
            low_asker=1,
        )
        accept = seller.buy_sell_response()
        assert accept is False, "Should reject when nobuysell > 0"


class TestTokensExhausted:
    """Tests for token exhaustion handling."""

    def test_buyer_no_bid_when_tokens_exhausted(self):
        """Buyer should return 0 when all tokens traded."""
        buyer = TruthTeller(
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
        seller = TruthTeller(
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
        buyer = TruthTeller(
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
        buyer = TruthTeller(
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
        seller = TruthTeller(
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
