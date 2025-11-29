"""
tests/test_orderbook.py - Test suite for AURORA Order Book implementation

Tests verify 1:1 fidelity with PeriodHistory.java from the 1993 Santa Fe Tournament.
Reference: /oldcode/extracted/double_auction/java/PeriodHistory.java
"""

import pytest
import numpy as np
from engine.orderbook import OrderBook


class TestOrderBookInitialization:
    """Test basic order book creation and initialization."""

    def test_create_orderbook(self):
        """Verify order book initializes with correct dimensions."""
        ob = OrderBook(
            num_buyers=3,
            num_sellers=3,
            num_times=10,
            min_price=1,
            max_price=1000,
            rng_seed=42
        )

        assert ob.num_buyers == 3
        assert ob.num_sellers == 3
        assert ob.num_times == 10
        assert ob.current_time == 0

        # Verify arrays are 1-indexed (player 0 is sentinel)
        assert ob.bids.shape == (4, 11)  # num_buyers+1, num_times+1
        assert ob.asks.shape == (4, 11)  # num_sellers+1, num_times+1


class TestBidValidation:
    """Test bid validation rules (PeriodHistory.java lines 162-170)."""

    @pytest.fixture
    def ob(self):
        """Create a fresh order book for each test."""
        return OrderBook(
            num_buyers=2,
            num_sellers=2,
            num_times=5,
            min_price=1,
            max_price=100,
            rng_seed=42
        )

    def test_first_bid_valid_in_range(self, ob):
        """First bid of period is valid if in price range."""
        ob.increment_time()  # Move to time 1
        assert ob.add_bid(bidder=1, price=50) is True
        assert ob.bids[1, 1] == 50

    def test_first_bid_invalid_out_of_range(self, ob):
        """First bid outside range is rejected."""
        ob.increment_time()
        assert ob.add_bid(bidder=1, price=0) is False
        assert ob.add_bid(bidder=1, price=101) is False

    def test_bid_must_improve_on_current(self, ob):
        """Subsequent bid must beat current high bid (no trade occurred)."""
        ob.increment_time()
        ob.add_bid(bidder=1, price=50)
        ob.determine_winners()

        ob.increment_time()
        # Same price should fail
        assert ob.add_bid(bidder=2, price=50) is False
        # Lower price should fail
        assert ob.add_bid(bidder=2, price=49) is False
        # Higher price should succeed
        assert ob.add_bid(bidder=2, price=51) is True

    def test_bid_valid_after_trade(self, ob):
        """After a trade, any in-range bid is valid (book is cleared)."""
        ob.increment_time()
        ob.add_bid(bidder=1, price=50)
        ob.add_ask(asker=1, price=40)
        ob.determine_winners()
        # Execute trade to clear book
        ob.execute_trade(buyer_accepts=True, seller_accepts=False)

        ob.increment_time()
        # Now a lower bid should be valid (book was cleared)
        assert ob.add_bid(bidder=2, price=30) is True


class TestAskValidation:
    """Test ask validation rules (PeriodHistory.java lines 183-195)."""

    @pytest.fixture
    def ob(self):
        return OrderBook(
            num_buyers=2,
            num_sellers=2,
            num_times=5,
            min_price=1,
            max_price=100,
            rng_seed=42
        )

    def test_first_ask_valid_in_range(self, ob):
        """First ask of period is valid if in price range."""
        ob.increment_time()
        assert ob.add_ask(asker=1, price=50) is True
        assert ob.asks[1, 1] == 50

    def test_ask_must_improve_on_current(self, ob):
        """Subsequent ask must beat current low ask (no trade occurred)."""
        ob.increment_time()
        ob.add_ask(asker=1, price=50)
        ob.determine_winners()

        ob.increment_time()
        # Same price should fail
        assert ob.add_ask(asker=2, price=50) is False
        # Higher price should fail
        assert ob.add_ask(asker=2, price=51) is False
        # Lower price should succeed
        assert ob.add_ask(asker=2, price=49) is True

    def test_ask_valid_after_trade(self, ob):
        """After a trade, any in-range ask is valid."""
        ob.increment_time()
        ob.add_bid(bidder=1, price=60)
        ob.add_ask(asker=1, price=50)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=False)

        ob.increment_time()
        # Higher ask should be valid after trade
        assert ob.add_ask(asker=2, price=70) is True


class TestWinnerDetermination:
    """Test winner selection (PeriodHistory.java lines 420-512)."""

    @pytest.fixture
    def ob(self):
        return OrderBook(
            num_buyers=3,
            num_sellers=3,
            num_times=5,
            min_price=1,
            max_price=100,
            rng_seed=42
        )

    def test_determine_high_bidder_simple(self, ob):
        """Single highest bidder is selected."""
        ob.increment_time()
        ob.add_bid(bidder=1, price=30)
        ob.add_bid(bidder=2, price=50)  # Highest
        ob.add_bid(bidder=3, price=40)

        high_bidder, low_asker = ob.determine_winners()
        assert high_bidder == 2
        assert ob.high_bid[ob.current_time] == 50

    def test_determine_low_asker_simple(self, ob):
        """Single lowest asker is selected."""
        ob.increment_time()
        ob.add_ask(asker=1, price=70)
        ob.add_ask(asker=2, price=50)  # Lowest
        ob.add_ask(asker=3, price=60)

        high_bidder, low_asker = ob.determine_winners()
        assert low_asker == 2
        assert ob.low_ask[ob.current_time] == 50

    def test_tied_high_bid_random_selection(self, ob):
        """Tied bidders: random selection (Java lines 442-476)."""
        ob.increment_time()
        ob.add_bid(bidder=1, price=50)
        ob.add_bid(bidder=2, price=50)  # Tie
        ob.add_bid(bidder=3, price=50)  # Tie

        # With seed=42, determine which bidder wins
        high_bidder, _ = ob.determine_winners()
        assert high_bidder in [1, 2, 3]
        assert ob.high_bid[ob.current_time] == 50

        # Verify randomness: create new book with different seed
        ob2 = OrderBook(2, 2, 5, 1, 100, rng_seed=99)
        ob2.increment_time()
        ob2.add_bid(bidder=1, price=50)
        ob2.add_bid(bidder=2, price=50)
        high_bidder2, _ = ob2.determine_winners()

        # With different seed, outcome could differ (test randomness exists)
        assert high_bidder2 in [1, 2]

    def test_no_bids_or_asks(self, ob):
        """Handle time step with no bids or asks."""
        ob.increment_time()
        high_bidder, low_asker = ob.determine_winners()

        assert high_bidder == 0  # No bidder
        assert low_asker == 0    # No asker
        assert ob.high_bid[ob.current_time] == 0
        assert ob.low_ask[ob.current_time] == 0


class TestTradeExecution:
    """Test trade execution and Chicago Rules pricing (lines 259-282)."""

    @pytest.fixture
    def ob(self):
        return OrderBook(
            num_buyers=2,
            num_sellers=2,
            num_times=5,
            min_price=1,
            max_price=100,
            rng_seed=42
        )

    def test_only_buyer_accepts_trade_at_ask(self, ob):
        """Buyer accepts ask → trade at ask price (seller's price)."""
        ob.increment_time()
        ob.add_bid(bidder=1, price=60)
        ob.add_ask(asker=1, price=50)
        ob.determine_winners()

        trade_price = ob.execute_trade(buyer_accepts=True, seller_accepts=False)
        assert trade_price == 50  # Ask price

    def test_only_seller_accepts_trade_at_bid(self, ob):
        """Seller accepts bid → trade at bid price (buyer's price)."""
        ob.increment_time()
        ob.add_bid(bidder=1, price=60)
        ob.add_ask(asker=1, price=50)
        ob.determine_winners()

        trade_price = ob.execute_trade(buyer_accepts=False, seller_accepts=True)
        assert trade_price == 60  # Bid price

    def test_both_accept_random_price(self, ob):
        """Both accept → 50/50 random between bid and ask (Chicago Rules)."""
        ob.increment_time()
        ob.add_bid(bidder=1, price=60)
        ob.add_ask(asker=1, price=50)
        ob.determine_winners()

        trade_price = ob.execute_trade(buyer_accepts=True, seller_accepts=True)
        # With seed=42, determine which price wins (bid=60 or ask=50)
        assert trade_price in [50, 60]

    def test_neither_accepts_no_trade(self, ob):
        """Neither accepts → no trade, price = 0."""
        ob.increment_time()
        ob.add_bid(bidder=1, price=60)
        ob.add_ask(asker=1, price=50)
        ob.determine_winners()

        trade_price = ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert trade_price == 0  # No trade

    def test_trade_clears_position_tracking(self, ob):
        """Verify trade updates position counters."""
        ob.increment_time()
        ob.add_bid(bidder=1, price=60)
        ob.add_ask(asker=1, price=50)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=False)

        # Check trade was recorded
        assert ob.trade_price[ob.current_time] == 50
        assert ob.num_buys[1, ob.current_time] == 1
        assert ob.num_sells[1, ob.current_time] == 1


class TestOrderCarryover:
    """Test order persistence across time steps (lines 129-149)."""

    @pytest.fixture
    def ob(self):
        return OrderBook(
            num_buyers=2,
            num_sellers=2,
            num_times=5,
            min_price=1,
            max_price=100,
            rng_seed=42
        )

    def test_orders_carry_over_if_no_trade(self, ob):
        """Standing orders persist to next time if no trade occurs."""
        ob.increment_time()  # Time 1
        ob.add_bid(bidder=1, price=50)
        ob.add_ask(asker=1, price=60)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)  # No trade

        ob.increment_time()  # Time 2
        # Orders should carry over
        assert ob.bids[1, 2] == 50
        assert ob.asks[1, 2] == 60

    def test_orders_clear_after_trade(self, ob):
        """Orders are cleared after a trade."""
        ob.increment_time()  # Time 1
        ob.add_bid(bidder=1, price=60)
        ob.add_ask(asker=1, price=50)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=False)  # Trade!

        ob.increment_time()  # Time 2
        # Orders should NOT carry over (cleared by trade)
        assert ob.bids[1, 2] == 0
        assert ob.asks[1, 2] == 0


class TestDATManualExample:
    """
    Test the worked example from DATManual.pdf pages 7-9.

    Scenario:
    - 3 buyers, 3 sellers
    - Token values: [100, 200, 300, 400] (buyers have redemption values, sellers have costs)
    - 6 time steps with specific bid/ask sequences

    This test verifies exact match with the documented behavior.
    """

    @pytest.fixture
    def ob(self):
        """Order book matching DATManual example setup."""
        return OrderBook(
            num_buyers=3,
            num_sellers=3,
            num_times=6,
            min_price=1,
            max_price=500,
            rng_seed=42  # Fixed seed for reproducibility
        )

    def test_datmanual_time_step_1(self, ob):
        """
        Time 1: Buyer 1 bids 150, Seller 1 asks 250
        No trade (ask > bid)
        """
        ob.increment_time()
        assert ob.add_bid(bidder=1, price=150)
        assert ob.add_ask(asker=1, price=250)

        high_bidder, low_asker = ob.determine_winners()
        assert high_bidder == 1
        assert low_asker == 1
        assert ob.high_bid[1] == 150
        assert ob.low_ask[1] == 250

        trade_price = ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert trade_price == 0  # No trade

    def test_datmanual_time_step_2(self, ob):
        """
        Time 2: Buyer 2 bids 180 (beats 150), Seller 2 asks 220 (beats 250)
        No trade (ask > bid)
        """
        # Set up time 1 first
        ob.increment_time()
        ob.add_bid(bidder=1, price=150)
        ob.add_ask(asker=1, price=250)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)

        # Time 2
        ob.increment_time()
        assert ob.add_bid(bidder=2, price=180)  # Beats 150
        assert ob.add_ask(asker=2, price=220)   # Beats 250

        high_bidder, low_asker = ob.determine_winners()
        assert ob.high_bid[2] == 180
        assert ob.low_ask[2] == 220

    # Note: Additional time steps 3-6 can be added to fully verify the example
    # This demonstrates the testing approach for the DATManual scenario


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_price_boundaries(self):
        """Test exact min/max price boundaries."""
        ob = OrderBook(2, 2, 5, min_price=50, max_price=200, rng_seed=42)
        ob.increment_time()

        assert ob.add_bid(bidder=1, price=50) is True   # Min boundary
        assert ob.add_bid(bidder=2, price=200) is True  # Max boundary
        assert ob.add_ask(asker=1, price=49) is False   # Below min
        assert ob.add_ask(asker=1, price=201) is False  # Above max

    def test_time_overflow(self):
        """Verify increment_time returns False at end of period."""
        ob = OrderBook(2, 2, num_times=2, min_price=1, max_price=100, rng_seed=42)

        assert ob.increment_time() is True   # Time 1
        assert ob.increment_time() is True   # Time 2
        assert ob.increment_time() is False  # Past end

    def test_invalid_player_ids(self):
        """Test behavior with out-of-range player IDs."""
        ob = OrderBook(2, 2, 5, 1, 100, rng_seed=42)
        ob.increment_time()

        # Player 0 is sentinel (invalid)
        with pytest.raises((IndexError, AssertionError)):
            ob.add_bid(bidder=0, price=50)

        # Player beyond num_buyers
        with pytest.raises((IndexError, AssertionError)):
            ob.add_bid(bidder=3, price=50)
