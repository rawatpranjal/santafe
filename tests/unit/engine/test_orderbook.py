# tests/unit/engine/test_orderbook.py
"""
Adversarial tests for OrderBook - the core AURORA trading logic.

These tests are designed to CATCH BUGS, not to pass.
Each test targets a specific invariant from the AURORA protocol.

Reference: DATManual_full_ocr.txt, PeriodHistory.java
"""

import pytest

from engine.orderbook import OrderBook


class TestBidImprovementRule:
    """
    AURORA Rule: New bid must STRICTLY exceed current high bid.
    Reference: DATManual lines 1200-1220
    """

    def test_bid_must_strictly_improve_high_bid(self):
        """A bid equal to high bid should be REJECTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()  # t=1

        # First bid establishes high bid
        assert ob.add_bid(1, 50) is True
        ob.determine_winners()
        assert ob.high_bid[ob.current_time] == 50

        ob.increment_time()  # t=2

        # Second bid at SAME price should be REJECTED
        result = ob.add_bid(2, 50)
        assert result is False, "Bid equal to high bid should be rejected"

    def test_bid_below_high_bid_rejected(self):
        """A bid below high bid should be REJECTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.determine_winners()
        ob.increment_time()

        result = ob.add_bid(2, 49)
        assert result is False, "Bid below high bid should be rejected"

    def test_bid_above_high_bid_accepted(self):
        """A bid strictly above high bid should be ACCEPTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.determine_winners()
        ob.increment_time()

        result = ob.add_bid(2, 51)
        assert result is True, "Bid above high bid should be accepted"

    def test_bid_after_trade_any_valid_price_accepted(self):
        """After a trade, book is cleared - any in-range bid valid."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        # Set up bid and ask
        ob.add_bid(1, 60)
        ob.add_ask(1, 50)
        ob.determine_winners()

        # Execute trade
        ob.execute_trade(buyer_accepts=True, seller_accepts=False)
        assert ob.trade_price[ob.current_time] == 50, "Trade should occur at ask"

        ob.increment_time()

        # After trade, even a low bid should be accepted
        result = ob.add_bid(2, 10)
        assert result is True, "After trade, any in-range bid should be accepted"


class TestAskImprovementRule:
    """
    AURORA Rule: New ask must STRICTLY undercut current low ask.
    Reference: DATManual lines 1220-1240
    """

    def test_ask_must_strictly_improve_low_ask(self):
        """An ask equal to low ask should be REJECTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        assert ob.add_ask(1, 50) is True
        ob.determine_winners()
        assert ob.low_ask[ob.current_time] == 50

        ob.increment_time()

        result = ob.add_ask(2, 50)
        assert result is False, "Ask equal to low ask should be rejected"

    def test_ask_above_low_ask_rejected(self):
        """An ask above low ask should be REJECTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_ask(1, 50)
        ob.determine_winners()
        ob.increment_time()

        result = ob.add_ask(2, 51)
        assert result is False, "Ask above low ask should be rejected"

    def test_ask_below_low_ask_accepted(self):
        """An ask strictly below low ask should be ACCEPTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_ask(1, 50)
        ob.determine_winners()
        ob.increment_time()

        result = ob.add_ask(2, 49)
        assert result is True, "Ask below low ask should be accepted"

    def test_ask_after_trade_any_valid_price_accepted(self):
        """After a trade, book is cleared - any in-range ask valid."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_bid(1, 60)
        ob.add_ask(1, 50)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=False)

        ob.increment_time()

        # After trade, even a high ask should be accepted
        result = ob.add_ask(2, 90)
        assert result is True, "After trade, any in-range ask should be accepted"


class TestPriceBounds:
    """
    AURORA Rule: All prices must be in [min_price, max_price].
    """

    def test_bid_below_min_price_rejected(self):
        """Bid below min_price should be REJECTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=10, max_price=100, rng_seed=42
        )
        ob.increment_time()

        result = ob.add_bid(1, 9)
        assert result is False, "Bid below min_price should be rejected"

    def test_bid_above_max_price_rejected(self):
        """Bid above max_price should be REJECTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        result = ob.add_bid(1, 101)
        assert result is False, "Bid above max_price should be rejected"

    def test_ask_below_min_price_rejected(self):
        """Ask below min_price should be REJECTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=10, max_price=100, rng_seed=42
        )
        ob.increment_time()

        result = ob.add_ask(1, 9)
        assert result is False, "Ask below min_price should be rejected"

    def test_ask_above_max_price_rejected(self):
        """Ask above max_price should be REJECTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        result = ob.add_ask(1, 101)
        assert result is False, "Ask above max_price should be rejected"

    def test_bid_at_boundary_accepted(self):
        """Bid exactly at min_price and max_price should be ACCEPTED."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=10, max_price=100, rng_seed=42
        )
        ob.increment_time()

        assert ob.add_bid(1, 10) is True, "Bid at min_price should be accepted"

        # Need to improve to add another bid
        ob.determine_winners()
        ob.increment_time()

        assert ob.add_bid(2, 100) is True, "Bid at max_price should be accepted"


class TestChicagoRulesTradeExecution:
    """
    Chicago Rules: Trade price determined by who accepts.
    - Buyer accepts -> trade at ask price
    - Seller accepts -> trade at bid price
    - Both accept -> random 50/50
    """

    def test_buyer_accepts_trade_at_ask_price(self):
        """When only buyer accepts, trade at ask price."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_bid(1, 60)
        ob.add_ask(1, 40)
        ob.determine_winners()

        price = ob.execute_trade(buyer_accepts=True, seller_accepts=False)
        assert price == 40, f"Buyer accepts -> trade at ask (40), got {price}"

    def test_seller_accepts_trade_at_bid_price(self):
        """When only seller accepts, trade at bid price."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_bid(1, 60)
        ob.add_ask(1, 40)
        ob.determine_winners()

        price = ob.execute_trade(buyer_accepts=False, seller_accepts=True)
        assert price == 60, f"Seller accepts -> trade at bid (60), got {price}"

    def test_neither_accepts_no_trade(self):
        """When neither accepts, no trade occurs."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_bid(1, 60)
        ob.add_ask(1, 40)
        ob.determine_winners()

        price = ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert price == 0, f"Neither accepts -> no trade (0), got {price}"

    def test_both_accept_random_price(self):
        """When both accept, price is randomly bid or ask."""
        bid_count = 0
        ask_count = 0

        for seed in range(100):
            ob = OrderBook(
                num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=seed
            )
            ob.increment_time()

            ob.add_bid(1, 60)
            ob.add_ask(1, 40)
            ob.determine_winners()

            price = ob.execute_trade(buyer_accepts=True, seller_accepts=True)
            assert price in (40, 60), f"Both accept -> price must be bid or ask, got {price}"

            if price == 60:
                bid_count += 1
            else:
                ask_count += 1

        # Should be roughly 50/50 (allow some variance)
        assert bid_count > 30, f"Both accept should sometimes trade at bid, got {bid_count}/100"
        assert ask_count > 30, f"Both accept should sometimes trade at ask, got {ask_count}/100"


class TestPositionTracking:
    """
    Position counters must be accurate and monotonically increasing.
    """

    def test_num_buys_increments_on_trade(self):
        """num_buys should increment when buyer trades."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        assert ob.num_buys[1, ob.current_time] == 0

        ob.add_bid(1, 60)
        ob.add_ask(1, 40)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=False, buyer_id=1, seller_id=1)

        assert ob.num_buys[1, ob.current_time] == 1, "num_buys should be 1 after trade"

    def test_num_sells_increments_on_trade(self):
        """num_sells should increment when seller trades."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        assert ob.num_sells[1, ob.current_time] == 0

        ob.add_bid(1, 60)
        ob.add_ask(1, 40)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=False, buyer_id=1, seller_id=1)

        assert ob.num_sells[1, ob.current_time] == 1, "num_sells should be 1 after trade"

    def test_positions_carry_forward_no_trade(self):
        """Position counters carry forward when no trade occurs."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_bid(1, 60)
        ob.add_ask(1, 40)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=False, buyer_id=1, seller_id=1)

        assert ob.num_buys[1, ob.current_time] == 1

        ob.increment_time()

        # No trade this step, but position should carry forward
        ob.add_bid(1, 30)  # Won't improve enough for trade
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)

        assert ob.num_buys[1, ob.current_time] == 1, "Position should carry forward"

    def test_positions_monotonically_increase(self):
        """Position counters should never decrease."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=20, min_price=1, max_price=100, rng_seed=42
        )

        prev_buys = 0
        prev_sells = 0

        for t in range(10):
            ob.increment_time()

            # Alternate trades
            ob.add_bid(1, 60 + t)
            ob.add_ask(1, 40 - t if 40 - t > 0 else 1)
            ob.determine_winners()

            if t % 2 == 0:
                ob.execute_trade(buyer_accepts=True, seller_accepts=False, buyer_id=1, seller_id=1)
            else:
                ob.execute_trade(buyer_accepts=False, seller_accepts=False)

            curr_buys = ob.num_buys[1, ob.current_time]
            curr_sells = ob.num_sells[1, ob.current_time]

            assert curr_buys >= prev_buys, f"num_buys decreased at t={t}"
            assert curr_sells >= prev_sells, f"num_sells decreased at t={t}"

            prev_buys = curr_buys
            prev_sells = curr_sells


class TestTieBreaking:
    """
    AURORA Rule: Ties are broken randomly.
    """

    def test_tied_bidders_randomly_selected(self):
        """When multiple bidders tie, winner is randomly selected."""
        winners = {1: 0, 2: 0}

        for seed in range(100):
            ob = OrderBook(
                num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=seed
            )
            ob.increment_time()

            # Both submit same bid in same time step
            ob.add_bid(1, 50)
            ob.add_bid(2, 50)
            ob.determine_winners()

            winner = ob.high_bidder[ob.current_time]
            assert winner in (1, 2), f"High bidder must be 1 or 2, got {winner}"
            winners[winner] += 1

        # Both should win sometimes
        assert winners[1] > 20, f"Bidder 1 should win sometimes, got {winners[1]}/100"
        assert winners[2] > 20, f"Bidder 2 should win sometimes, got {winners[2]}/100"

    def test_tied_askers_randomly_selected(self):
        """When multiple askers tie, winner is randomly selected."""
        winners = {1: 0, 2: 0}

        for seed in range(100):
            ob = OrderBook(
                num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=seed
            )
            ob.increment_time()

            ob.add_ask(1, 50)
            ob.add_ask(2, 50)
            ob.determine_winners()

            winner = ob.low_asker[ob.current_time]
            assert winner in (1, 2), f"Low asker must be 1 or 2, got {winner}"
            winners[winner] += 1

        assert winners[1] > 20, f"Asker 1 should win sometimes, got {winners[1]}/100"
        assert winners[2] > 20, f"Asker 2 should win sometimes, got {winners[2]}/100"


class TestOneIndexing:
    """
    AURORA uses 1-indexed player IDs and time steps.
    """

    def test_player_ids_are_1_indexed(self):
        """Player IDs start at 1, not 0."""
        ob = OrderBook(
            num_buyers=3, num_sellers=3, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        # Should accept player IDs 1, 2, 3
        assert ob.add_bid(1, 50) is True
        assert ob.add_bid(2, 51) is True
        assert ob.add_bid(3, 52) is True

        # Player ID 0 should raise assertion
        with pytest.raises(AssertionError):
            ob.add_bid(0, 53)

    def test_time_starts_at_1_after_increment(self):
        """First increment_time() sets current_time to 1."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )

        assert ob.current_time == 0, "Initial time should be 0"
        ob.increment_time()
        assert ob.current_time == 1, "After first increment, time should be 1"


class TestDeadsteps:
    """
    AURORA Rule: Period ends early after N consecutive no-trade steps.
    """

    def test_deadsteps_triggers_early_termination(self):
        """should_terminate_early() returns True after deadsteps no-trades."""
        ob = OrderBook(
            num_buyers=2,
            num_sellers=2,
            num_times=100,
            min_price=1,
            max_price=100,
            rng_seed=42,
            deadsteps=3,
        )

        ob.increment_time()
        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        # No trade 1
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert ob.should_terminate_early() is False

        ob.increment_time()
        # No trade 2
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert ob.should_terminate_early() is False

        ob.increment_time()
        # No trade 3 - should trigger
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert ob.should_terminate_early() is True, "Should terminate after 3 no-trades"

    def test_trade_resets_deadsteps_counter(self):
        """A successful trade resets the consecutive no-trade counter."""
        ob = OrderBook(
            num_buyers=2,
            num_sellers=2,
            num_times=100,
            min_price=1,
            max_price=100,
            rng_seed=42,
            deadsteps=3,
        )

        ob.increment_time()
        ob.add_bid(1, 60)
        ob.add_ask(1, 50)
        ob.determine_winners()

        # No trade 1
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        ob.increment_time()

        # No trade 2
        ob.add_bid(1, 61)
        ob.add_ask(1, 49)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        ob.increment_time()

        # Trade! Should reset counter
        ob.add_bid(1, 62)
        ob.add_ask(1, 48)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=False)
        assert ob.consecutive_no_trades == 0, "Trade should reset counter"

        ob.increment_time()
        # No trade 1 (restarted)
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert ob.should_terminate_early() is False

    def test_deadsteps_zero_disables_early_termination(self):
        """deadsteps=0 means never terminate early."""
        ob = OrderBook(
            num_buyers=2,
            num_sellers=2,
            num_times=100,
            min_price=1,
            max_price=100,
            rng_seed=42,
            deadsteps=0,  # Disabled
        )

        for _ in range(50):
            ob.increment_time()
            ob.execute_trade(buyer_accepts=False, seller_accepts=False)
            assert ob.should_terminate_early() is False


class TestBookClearing:
    """
    After a trade, the order book should be cleared.
    """

    def test_book_cleared_after_trade(self):
        """Standing orders are cleared after a trade."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_bid(1, 60)
        ob.add_ask(1, 40)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=False)

        ob.increment_time()

        # After trade, bids/asks should be 0 (cleared)
        assert ob.bids[1, ob.current_time] == 0, "Bids should be cleared after trade"
        assert ob.asks[1, ob.current_time] == 0, "Asks should be cleared after trade"

    def test_orders_carry_forward_without_trade(self):
        """Standing orders carry forward when no trade occurs."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)  # No trade

        ob.increment_time()

        # Orders should carry forward
        assert ob.bids[1, ob.current_time] == 50, "Bid should carry forward"
        assert ob.asks[1, ob.current_time] == 60, "Ask should carry forward"


class TestFirstStepBehavior:
    """
    First time step has special rules - no previous state to improve upon.
    """

    def test_first_step_any_valid_bid_accepted(self):
        """At t=1, any in-range bid should be accepted."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        # All in-range bids should be accepted at first step
        assert ob.add_bid(1, 1) is True
        assert ob.add_bid(2, 100) is True

    def test_first_step_any_valid_ask_accepted(self):
        """At t=1, any in-range ask should be accepted."""
        ob = OrderBook(
            num_buyers=2, num_sellers=2, num_times=10, min_price=1, max_price=100, rng_seed=42
        )
        ob.increment_time()

        assert ob.add_ask(1, 1) is True
        assert ob.add_ask(2, 100) is True
