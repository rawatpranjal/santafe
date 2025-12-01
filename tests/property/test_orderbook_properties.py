# tests/property/test_orderbook_properties.py
"""
Property-based tests for OrderBook invariants using Hypothesis.

These tests verify that key invariants hold across a wide range of inputs,
catching edge cases that example-based tests might miss.
"""

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from engine.orderbook import OrderBook

# =============================================================================
# Strategies for generating test data
# =============================================================================


@st.composite
def orderbook_params(draw):
    """Generate valid OrderBook initialization parameters."""
    num_buyers = draw(st.integers(min_value=1, max_value=10))
    num_sellers = draw(st.integers(min_value=1, max_value=10))
    num_times = draw(st.integers(min_value=10, max_value=200))
    min_price = draw(st.integers(min_value=0, max_value=50))
    max_price = draw(st.integers(min_value=min_price + 50, max_value=500))
    rng_seed = draw(st.integers(min_value=0, max_value=100000))

    return {
        "num_buyers": num_buyers,
        "num_sellers": num_sellers,
        "num_times": num_times,
        "min_price": min_price,
        "max_price": max_price,
        "rng_seed": rng_seed,
    }


@st.composite
def valid_bid(draw, min_price, max_price):
    """Generate a valid bid price."""
    return draw(st.integers(min_value=min_price, max_value=max_price))


@st.composite
def valid_ask(draw, min_price, max_price):
    """Generate a valid ask price."""
    return draw(st.integers(min_value=min_price, max_value=max_price))


# =============================================================================
# Property Tests: OrderBook Invariants
# =============================================================================


class TestOrderBookInvariants:
    """Property tests for OrderBook invariants."""

    @given(orderbook_params())
    @settings(max_examples=50)
    def test_initialization_invariants(self, params):
        """Newly created OrderBook should satisfy all invariants."""
        ob = OrderBook(**params)

        # 1. Current time starts at 0
        assert ob.current_time == 0

        # 2. No initial bids or asks
        assert ob.high_bid[0] == 0
        assert ob.low_ask[0] == 0

        # 3. No initial trades
        assert ob.trade_price[0] == 0

        # 4. Arrays have correct shapes
        assert ob.bids.shape == (params["num_buyers"] + 1, params["num_times"] + 1)
        assert ob.asks.shape == (params["num_sellers"] + 1, params["num_times"] + 1)

    @given(orderbook_params())
    @settings(max_examples=50)
    def test_time_advances_monotonically(self, params):
        """Time should only advance, never decrease."""
        ob = OrderBook(**params)

        previous_time = ob.current_time
        for _ in range(min(10, params["num_times"])):
            ob.increment_time()
            assert ob.current_time > previous_time
            previous_time = ob.current_time

    @given(orderbook_params())
    @settings(max_examples=30)
    def test_bid_validation_rejects_out_of_range(self, params):
        """Bids outside valid range should be rejected."""
        ob = OrderBook(**params)
        ob.increment_time()

        buyer_id = 1

        # Bid below minimum should be rejected
        below_min = params["min_price"] - 1
        result = ob.add_bid(buyer_id, below_min)
        assert result is False

        # Bid above maximum should be rejected
        above_max = params["max_price"] + 1
        result = ob.add_bid(buyer_id, above_max)
        assert result is False

    @given(orderbook_params())
    @settings(max_examples=30)
    def test_ask_validation_rejects_out_of_range(self, params):
        """Asks outside valid range should be rejected."""
        ob = OrderBook(**params)
        ob.increment_time()

        seller_id = 1

        # Ask below minimum should be rejected
        below_min = params["min_price"] - 1
        result = ob.add_ask(seller_id, below_min)
        assert result is False

        # Ask above maximum should be rejected
        above_max = params["max_price"] + 1
        result = ob.add_ask(seller_id, above_max)
        assert result is False

    @given(orderbook_params())
    @settings(max_examples=30)
    def test_trade_price_in_valid_range(self, params):
        """Trade prices should be within [min_price, max_price]."""
        ob = OrderBook(**params)

        # Set up a trade scenario
        ob.increment_time()

        bid_price = params["min_price"] + 50
        ask_price = params["min_price"] + 40

        assume(bid_price <= params["max_price"])
        assume(ask_price >= params["min_price"])
        assume(bid_price >= ask_price)  # Valid trade scenario

        ob.add_bid(1, bid_price)
        ob.add_ask(1, ask_price)

        # Execute trade
        ob.execute_trade(1, 1, ask_price)  # Trade at ask price

        # Verify trade price is in range
        t = ob.current_time
        if ob.trade_price[t] > 0:
            assert params["min_price"] <= ob.trade_price[t] <= params["max_price"]

    @given(orderbook_params())
    @settings(max_examples=30)
    def test_position_counts_never_decrease(self, params):
        """num_buys and num_sells should be monotonically non-decreasing."""
        ob = OrderBook(**params)

        buyer_id = 1
        seller_id = 1

        # Execute some trades
        for _ in range(min(3, params["num_times"] - 1)):
            ob.increment_time()
            t = ob.current_time

            # Get previous counts
            prev_buys = ob.num_buys[buyer_id, t - 1] if t > 1 else 0
            prev_sells = ob.num_sells[seller_id, t - 1] if t > 1 else 0

            # Submit and execute trade
            bid = params["min_price"] + 30
            ask = params["min_price"] + 25
            assume(bid <= params["max_price"])
            assume(ask >= params["min_price"])

            ob.add_bid(buyer_id, bid)
            ob.add_ask(seller_id, ask)
            ob.execute_trade(buyer_id, seller_id, ask)

            # Verify monotonicity
            assert ob.num_buys[buyer_id, t] >= prev_buys
            assert ob.num_sells[seller_id, t] >= prev_sells


# =============================================================================
# Property Tests: Chicago Rules
# =============================================================================


class TestChicagoRulesProperties:
    """Property tests for Chicago Rules pricing."""

    @given(orderbook_params())
    @settings(max_examples=30)
    def test_trade_records_price(self, params):
        """Trade should record the correct price when buyer accepts."""
        ob = OrderBook(**params)
        ob.increment_time()

        # Set up trade with valid prices
        bid = params["min_price"] + 30
        ask = params["min_price"] + 40
        assume(bid <= params["max_price"])
        assume(ask <= params["max_price"])
        assume(ask >= params["min_price"])

        # Add bid and ask
        ob.add_bid(1, bid)
        ob.add_ask(1, ask)

        # Determine winners to set high_bid/low_ask
        ob.determine_winners()

        t = ob.current_time

        # Buyer accepts seller's ask (trade at ask price)
        price = ob.execute_trade(
            buyer_accepts=True,
            seller_accepts=False,
            buyer_id=1,
            seller_id=1,
        )

        # Trade price should be the ask (seller's price)
        assert ob.trade_price[t] == ask
        assert price == ask


# =============================================================================
# Property Tests: Agent Position Tracking
# =============================================================================


class TestPositionTrackingProperties:
    """Property tests for agent position tracking."""

    @given(orderbook_params())
    @settings(max_examples=30)
    def test_buyer_trades_equal_seller_trades(self, params):
        """Total buyer trades should equal total seller trades."""
        ob = OrderBook(**params)

        # Execute some trades
        num_trades = min(3, params["num_times"] - 1, params["num_buyers"], params["num_sellers"])

        for i in range(num_trades):
            ob.increment_time()

            buyer_id = (i % params["num_buyers"]) + 1
            seller_id = (i % params["num_sellers"]) + 1

            bid = params["min_price"] + 50
            ask = params["min_price"] + 40
            assume(bid <= params["max_price"])
            assume(ask >= params["min_price"])

            ob.add_bid(buyer_id, bid)
            ob.add_ask(seller_id, ask)
            ob.execute_trade(buyer_id, seller_id, ask)

        # Sum all buys and sells at final time
        t = ob.current_time
        total_buys = sum(ob.num_buys[b, t] for b in range(1, params["num_buyers"] + 1))
        total_sells = sum(ob.num_sells[s, t] for s in range(1, params["num_sellers"] + 1))

        assert (
            total_buys == total_sells
        ), f"Buyer trades ({total_buys}) != seller trades ({total_sells})"
