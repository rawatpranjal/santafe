"""
Phase 1 Task 1.1: Order Book & Token Generation Tests

This module validates the AURORA order book mechanics and token generation
as specified in PLAN.md.

References:
- engine/orderbook.py - OrderBook class (port of PeriodHistory.java)
- engine/token_generator.py - TokenGenerator class
- reference/oldcode/DATManual_full_ocr.txt - AURORA protocol spec
"""

import pytest
import numpy as np
from collections import Counter

from engine.orderbook import OrderBook
from engine.token_generator import TokenGenerator, generate_tokens


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def basic_orderbook() -> OrderBook:
    """Create a basic 2x2 orderbook for testing."""
    return OrderBook(
        num_buyers=2,
        num_sellers=2,
        num_times=10,
        min_price=1,
        max_price=100,
        rng_seed=42,
    )


@pytest.fixture
def large_orderbook() -> OrderBook:
    """Create an 8x8 orderbook for tournament-scale testing."""
    return OrderBook(
        num_buyers=8,
        num_sellers=8,
        num_times=100,
        min_price=1,
        max_price=100,
        rng_seed=42,
    )


# =============================================================================
# TEST CLASS: OrderBook Initialization
# =============================================================================


class TestOrderBookInitialization:
    """Tests for OrderBook array dimensions and initialization."""

    def test_orderbook_creates_correct_dimensions(self, basic_orderbook: OrderBook) -> None:
        """Arrays must be sized (num_players+1, num_times+1) for 1-indexing."""
        ob = basic_orderbook

        # Bids array: (num_buyers+1, num_times+1)
        assert ob.bids.shape == (3, 11), f"Expected (3, 11), got {ob.bids.shape}"

        # Asks array: (num_sellers+1, num_times+1)
        assert ob.asks.shape == (3, 11), f"Expected (3, 11), got {ob.asks.shape}"

        # Position arrays
        assert ob.num_buys.shape == (3, 11)
        assert ob.num_sells.shape == (3, 11)

        # Market state arrays (1D, time-indexed)
        assert ob.high_bid.shape == (11,)
        assert ob.low_ask.shape == (11,)
        assert ob.high_bidder.shape == (11,)
        assert ob.low_asker.shape == (11,)
        assert ob.trade_price.shape == (11,)

    def test_orderbook_uses_integer_math(self, basic_orderbook: OrderBook) -> None:
        """All price arrays must use integer dtype (AURORA Strict Mode)."""
        ob = basic_orderbook

        assert ob.bids.dtype == np.int32, f"bids dtype: {ob.bids.dtype}"
        assert ob.asks.dtype == np.int32, f"asks dtype: {ob.asks.dtype}"
        assert ob.high_bid.dtype == np.int32
        assert ob.low_ask.dtype == np.int32
        assert ob.trade_price.dtype == np.int32

    def test_orderbook_initializes_to_zero(self, basic_orderbook: OrderBook) -> None:
        """All arrays should initialize to zero."""
        ob = basic_orderbook

        assert np.all(ob.bids == 0)
        assert np.all(ob.asks == 0)
        assert np.all(ob.high_bid == 0)
        assert np.all(ob.low_ask == 0)
        assert np.all(ob.trade_price == 0)
        assert ob.current_time == 0


# =============================================================================
# TEST CLASS: Bid Validation
# =============================================================================


class TestBidValidation:
    """Tests for bid submission validation (AURORA rules)."""

    def test_first_bid_accepted_in_range(self, basic_orderbook: OrderBook) -> None:
        """Any bid in [min_price, max_price] should be accepted as first bid."""
        ob = basic_orderbook
        ob.increment_time()  # Move to t=1

        # First bid at min_price
        assert ob.add_bid(1, 1) is True

        # Create new orderbook for fresh test
        ob2 = OrderBook(2, 2, 10, 1, 100, 42)
        ob2.increment_time()

        # First bid at max_price
        assert ob2.add_bid(1, 100) is True

        # First bid at mid-range
        ob3 = OrderBook(2, 2, 10, 1, 100, 42)
        ob3.increment_time()
        assert ob3.add_bid(1, 50) is True

    def test_bid_below_min_rejected(self, basic_orderbook: OrderBook) -> None:
        """Bid < min_price must be rejected."""
        ob = basic_orderbook
        ob.increment_time()

        # Try bid below min_price (0 when min=1)
        assert ob.add_bid(1, 0) is False
        # Verify no bid was recorded
        assert ob.bids[1, 1] == 0

    def test_bid_above_max_rejected(self, basic_orderbook: OrderBook) -> None:
        """Bid > max_price must be rejected."""
        ob = basic_orderbook
        ob.increment_time()

        # Try bid above max_price (101 when max=100)
        assert ob.add_bid(1, 101) is False
        assert ob.bids[1, 1] == 0

    def test_bid_must_improve_when_no_trade(self, basic_orderbook: OrderBook) -> None:
        """Bid must strictly improve high_bid when no previous trade."""
        ob = basic_orderbook
        ob.increment_time()  # t=1

        # First bid at 50
        ob.add_bid(1, 50)
        ob.determine_winners()

        ob.increment_time()  # t=2, no trade

        # Bid at same price should be rejected
        assert ob.add_bid(2, 50) is False

        # Bid lower should be rejected
        assert ob.add_bid(2, 49) is False

        # Bid higher should be accepted
        assert ob.add_bid(2, 51) is True

    def test_bid_resets_after_trade(self, basic_orderbook: OrderBook) -> None:
        """Any in-range bid should be valid after a trade (book cleared)."""
        ob = basic_orderbook
        ob.increment_time()  # t=1

        # Setup: bid at 50, ask at 60
        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        # Execute trade (both accept)
        ob.execute_trade(buyer_accepts=True, seller_accepts=True)

        ob.increment_time()  # t=2, trade occurred at t=1

        # After trade, any in-range bid is valid (book cleared)
        # Even a bid lower than previous high_bid should work
        assert ob.add_bid(1, 30) is True


# =============================================================================
# TEST CLASS: Ask Validation
# =============================================================================


class TestAskValidation:
    """Tests for ask submission validation (AURORA rules)."""

    def test_first_ask_accepted_in_range(self, basic_orderbook: OrderBook) -> None:
        """Any ask in [min_price, max_price] should be accepted as first ask."""
        ob = basic_orderbook
        ob.increment_time()

        assert ob.add_ask(1, 1) is True

        ob2 = OrderBook(2, 2, 10, 1, 100, 42)
        ob2.increment_time()
        assert ob2.add_ask(1, 100) is True

        ob3 = OrderBook(2, 2, 10, 1, 100, 42)
        ob3.increment_time()
        assert ob3.add_ask(1, 50) is True

    def test_ask_below_min_rejected(self, basic_orderbook: OrderBook) -> None:
        """Ask < min_price must be rejected."""
        ob = basic_orderbook
        ob.increment_time()

        assert ob.add_ask(1, 0) is False
        assert ob.asks[1, 1] == 0

    def test_ask_above_max_rejected(self, basic_orderbook: OrderBook) -> None:
        """Ask > max_price must be rejected."""
        ob = basic_orderbook
        ob.increment_time()

        assert ob.add_ask(1, 101) is False
        assert ob.asks[1, 1] == 0

    def test_ask_must_improve_when_no_trade(self, basic_orderbook: OrderBook) -> None:
        """Ask must strictly improve (be lower than) low_ask when no previous trade."""
        ob = basic_orderbook
        ob.increment_time()  # t=1

        # First ask at 60
        ob.add_ask(1, 60)
        ob.determine_winners()

        ob.increment_time()  # t=2, no trade

        # Ask at same price should be rejected
        assert ob.add_ask(2, 60) is False

        # Ask higher should be rejected
        assert ob.add_ask(2, 61) is False

        # Ask lower should be accepted
        assert ob.add_ask(2, 59) is True

    def test_ask_resets_after_trade(self, basic_orderbook: OrderBook) -> None:
        """Any in-range ask should be valid after a trade (book cleared)."""
        ob = basic_orderbook
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=True)

        ob.increment_time()  # Trade occurred at t=1

        # After trade, any in-range ask is valid
        assert ob.add_ask(1, 90) is True  # Higher than previous low_ask


# =============================================================================
# TEST CLASS: Crossed Market Handling (AURORA Two-Phase Protocol)
# =============================================================================


class TestCrossedMarketHandling:
    """
    Tests for AURORA crossed market behavior.

    In AURORA, crossing bids/asks ARE ALLOWED because:
    1. Bid-offer phase: Quotes are submitted (crossing creates trade opportunity)
    2. Buy-sell phase: Trade only happens when BOTH parties agree

    This is different from continuous double auctions where crossing
    triggers automatic execution. AURORA's two-phase design allows
    agents to observe the crossed state before deciding to trade.
    """

    def test_bid_accepted_when_crosses_standing_ask(self, basic_orderbook: OrderBook) -> None:
        """Bid >= low_ask is ACCEPTED in AURORA (creates trade opportunity)."""
        ob = basic_orderbook
        ob.increment_time()

        # First, establish a standing ask at 50
        ob.add_ask(1, 50)
        ob.determine_winners()

        ob.increment_time()  # t=2, orders carry over

        # In AURORA, crossing bids ARE accepted (but must improve)
        # Bid below standing ask (improves, not crossing) - accepted
        assert ob.add_bid(1, 49) is True

        ob.determine_winners()
        ob.increment_time()

        # Bid that crosses (>= ask) and improves is also accepted
        assert ob.add_bid(2, 50) is True  # Crosses but improves

    def test_ask_accepted_when_crosses_standing_bid(self, basic_orderbook: OrderBook) -> None:
        """Ask <= high_bid is ACCEPTED in AURORA (creates trade opportunity)."""
        ob = basic_orderbook
        ob.increment_time()

        # First, establish a standing bid at 50
        ob.add_bid(1, 50)
        ob.determine_winners()

        ob.increment_time()  # t=2, orders carry over

        # In AURORA, crossing asks ARE accepted (but must improve)
        # Ask above standing bid (improves, not crossing) - accepted
        assert ob.add_ask(1, 51) is True

        ob.determine_winners()
        ob.increment_time()

        # Ask that crosses (<= bid) and improves is also accepted
        assert ob.add_ask(2, 50) is True  # Crosses but improves

    def test_crossed_market_does_not_execute_automatically(self, basic_orderbook: OrderBook) -> None:
        """Crossing creates trade opportunity but no automatic execution."""
        ob = basic_orderbook
        ob.increment_time()

        # Setup standing orders that create a narrow spread
        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        ob.increment_time()

        # Improved orders that cross
        ob.add_bid(2, 55)  # Above previous high_bid
        ob.add_ask(2, 55)  # Below previous low_ask, equals bid = crossed

        # Both should be accepted
        ob.determine_winners()

        # But NO automatic trade execution (trade happens in buy-sell phase)
        assert ob.trade_price[ob.current_time] == 0

        # Both orders recorded
        assert ob.high_bid[ob.current_time] == 55
        assert ob.low_ask[ob.current_time] == 55


# =============================================================================
# TEST CLASS: Winner Determination
# =============================================================================


class TestWinnerDetermination:
    """Tests for high bidder and low asker selection."""

    def test_highest_bidder_wins(self, basic_orderbook: OrderBook) -> None:
        """high_bidder should be the buyer with highest bid."""
        ob = basic_orderbook
        ob.increment_time()

        ob.add_bid(1, 40)  # Lower bid
        ob.add_bid(2, 50)  # Higher bid

        ob.determine_winners()

        assert ob.high_bid[1] == 50
        assert ob.high_bidder[1] == 2

    def test_lowest_asker_wins(self, basic_orderbook: OrderBook) -> None:
        """low_asker should be the seller with lowest ask."""
        ob = basic_orderbook
        ob.increment_time()

        ob.add_ask(1, 70)  # Higher ask
        ob.add_ask(2, 60)  # Lower ask

        ob.determine_winners()

        assert ob.low_ask[1] == 60
        assert ob.low_asker[1] == 2

    def test_tied_bids_random_selection(self) -> None:
        """Statistical test: tied bids should be resolved ~50/50."""
        wins_by_bidder: Counter[int] = Counter()
        trials = 500

        for seed in range(trials):
            ob = OrderBook(2, 2, 10, 1, 100, seed)
            ob.increment_time()

            # Both bidders bid same price
            ob.add_bid(1, 50)
            ob.add_bid(2, 50)

            ob.determine_winners()
            wins_by_bidder[ob.high_bidder[1]] += 1

        # Expect roughly 50% each (within 10% tolerance)
        ratio = wins_by_bidder[1] / trials
        assert 0.40 <= ratio <= 0.60, f"Bidder 1 won {ratio*100:.1f}%, expected ~50%"

    def test_tied_asks_random_selection(self) -> None:
        """Statistical test: tied asks should be resolved ~50/50."""
        wins_by_asker: Counter[int] = Counter()
        trials = 500

        for seed in range(trials):
            ob = OrderBook(2, 2, 10, 1, 100, seed)
            ob.increment_time()

            # Both sellers ask same price
            ob.add_ask(1, 60)
            ob.add_ask(2, 60)

            ob.determine_winners()
            wins_by_asker[ob.low_asker[1]] += 1

        ratio = wins_by_asker[1] / trials
        assert 0.40 <= ratio <= 0.60, f"Asker 1 won {ratio*100:.1f}%, expected ~50%"

    def test_status_codes_correct(self, basic_orderbook: OrderBook) -> None:
        """Status codes 0-4 should be assigned correctly."""
        ob = basic_orderbook
        ob.increment_time()

        # Bidder 1 wins, Bidder 2 loses
        ob.add_bid(1, 50)
        ob.add_bid(2, 40)

        ob.determine_winners()

        # Status 2 = new bid, now current winner
        # Status 3 = new bid, beaten by another
        assert ob.bid_status[1] == 2, f"Winner should have status 2, got {ob.bid_status[1]}"
        assert ob.bid_status[2] == 3, f"Loser should have status 3, got {ob.bid_status[2]}"


# =============================================================================
# TEST CLASS: Chicago Rules Pricing
# =============================================================================


class TestChicagoRulesPricing:
    """Tests for Chicago Rules trade pricing."""

    def test_buyer_only_accepts_trades_at_ask(self, basic_orderbook: OrderBook) -> None:
        """If only buyer accepts, trade at ask price (seller's price)."""
        ob = basic_orderbook
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        price = ob.execute_trade(buyer_accepts=True, seller_accepts=False)

        assert price == 60, f"Expected ask price 60, got {price}"

    def test_seller_only_accepts_trades_at_bid(self, basic_orderbook: OrderBook) -> None:
        """If only seller accepts, trade at bid price (buyer's price)."""
        ob = basic_orderbook
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        price = ob.execute_trade(buyer_accepts=False, seller_accepts=True)

        assert price == 50, f"Expected bid price 50, got {price}"

    def test_both_accept_fifty_fifty_random(self) -> None:
        """Statistical test: both accept → 50/50 between bid and ask."""
        bid_prices = 0
        ask_prices = 0
        trials = 1000

        for seed in range(trials):
            ob = OrderBook(2, 2, 10, 1, 100, seed)
            ob.increment_time()

            ob.add_bid(1, 50)
            ob.add_ask(1, 60)
            ob.determine_winners()

            price = ob.execute_trade(buyer_accepts=True, seller_accepts=True)

            if price == 50:
                bid_prices += 1
            elif price == 60:
                ask_prices += 1

        bid_ratio = bid_prices / trials
        assert 0.45 <= bid_ratio <= 0.55, f"Bid price ratio {bid_ratio*100:.1f}%, expected ~50%"

    def test_neither_accepts_no_trade(self, basic_orderbook: OrderBook) -> None:
        """If neither accepts, no trade (price = 0)."""
        ob = basic_orderbook
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        price = ob.execute_trade(buyer_accepts=False, seller_accepts=False)

        assert price == 0, f"Expected no trade (price=0), got {price}"


# =============================================================================
# TEST CLASS: Order Carryover
# =============================================================================


class TestOrderCarryover:
    """Tests for order persistence across time steps."""

    def test_orders_persist_when_no_trade(self, basic_orderbook: OrderBook) -> None:
        """Standing orders should carry over when no trade occurs."""
        ob = basic_orderbook
        ob.increment_time()  # t=1

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        # No trade
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)

        ob.increment_time()  # t=2

        # Orders should carry over
        assert ob.bids[1, 2] == 50, "Bid should carry over"
        assert ob.asks[1, 2] == 60, "Ask should carry over"

    def test_orders_clear_when_trade_occurs(self, basic_orderbook: OrderBook) -> None:
        """Orders should be cleared (reset to 0) after a trade."""
        ob = basic_orderbook
        ob.increment_time()  # t=1

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        # Trade occurs
        ob.execute_trade(buyer_accepts=True, seller_accepts=True)

        ob.increment_time()  # t=2

        # Orders should be cleared
        assert ob.bids[1, 2] == 0, "Bid should be cleared after trade"
        assert ob.asks[1, 2] == 0, "Ask should be cleared after trade"

    def test_positions_always_accumulate(self, basic_orderbook: OrderBook) -> None:
        """num_buys/num_sells should never reset within a period."""
        ob = basic_orderbook

        # First trade
        ob.increment_time()  # t=1
        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=True)

        assert ob.num_buys[1, 1] == 1
        assert ob.num_sells[1, 1] == 1

        # Second trade
        ob.increment_time()  # t=2
        ob.add_bid(1, 55)
        ob.add_ask(1, 65)
        ob.determine_winners()
        ob.execute_trade(buyer_accepts=True, seller_accepts=True)

        # Positions should accumulate, not reset
        assert ob.num_buys[1, 2] == 2, "Buys should accumulate"
        assert ob.num_sells[1, 2] == 2, "Sells should accumulate"


# =============================================================================
# TEST CLASS: Token Generation
# =============================================================================


class TestTokenGeneration:
    """Tests for token (valuation/cost) generation."""

    def test_game_type_6453_produces_expected_values(self) -> None:
        """Reference seed 6453 should produce consistent values."""
        gen = TokenGenerator(game_type=6453, num_tokens=4, seed=42)
        gen.new_round()

        buyer_tokens = gen.generate_tokens(is_buyer=True)
        seller_tokens = gen.generate_tokens(is_buyer=False)

        # Values should be non-negative integers
        assert all(isinstance(t, int) for t in buyer_tokens)
        assert all(isinstance(t, int) for t in seller_tokens)
        assert all(t >= 0 for t in buyer_tokens)
        assert all(t >= 0 for t in seller_tokens)

        # Should have num_tokens values
        assert len(buyer_tokens) == 4
        assert len(seller_tokens) == 4

    def test_buyer_tokens_sorted_descending(self) -> None:
        """Buyer valuations should be sorted high to low."""
        gen = TokenGenerator(game_type=6453, num_tokens=4, seed=12345)
        gen.new_round()

        tokens = gen.generate_tokens(is_buyer=True)

        # Check descending order
        for i in range(len(tokens) - 1):
            assert tokens[i] >= tokens[i + 1], \
                f"Buyer tokens not descending: {tokens}"

    def test_seller_tokens_sorted_ascending(self) -> None:
        """Seller costs should be sorted low to high."""
        gen = TokenGenerator(game_type=6453, num_tokens=4, seed=12345)
        gen.new_round()

        tokens = gen.generate_tokens(is_buyer=False)

        # Check ascending order
        for i in range(len(tokens) - 1):
            assert tokens[i] <= tokens[i + 1], \
                f"Seller tokens not ascending: {tokens}"

    def test_token_generation_reproducible(self) -> None:
        """Same seed should produce same tokens."""
        gen1 = TokenGenerator(game_type=6453, num_tokens=4, seed=42)
        gen1.new_round()
        buyer1 = gen1.generate_tokens(is_buyer=True)
        seller1 = gen1.generate_tokens(is_buyer=False)

        gen2 = TokenGenerator(game_type=6453, num_tokens=4, seed=42)
        gen2.new_round()
        buyer2 = gen2.generate_tokens(is_buyer=True)
        seller2 = gen2.generate_tokens(is_buyer=False)

        assert buyer1 == buyer2, "Buyer tokens not reproducible"
        assert seller1 == seller2, "Seller tokens not reproducible"

    def test_generate_tokens_wrapper_function(self) -> None:
        """Test the convenience wrapper function."""
        buyer_vals, seller_costs = generate_tokens(
            num_buyers=2,
            num_sellers=2,
            num_tokens=4,
            price_min=0,
            price_max=100,
            game_type=6453,
            seed=42,
        )

        assert len(buyer_vals) == 2
        assert len(seller_costs) == 2
        assert len(buyer_vals[0]) == 4
        assert len(seller_costs[0]) == 4

    def test_game_type_weights_calculation(self) -> None:
        """Verify weight calculation from game_type digits."""
        # Game type 6453:
        # digit 1 = 6 -> w[1] = 3^6 - 1 = 728
        # digit 2 = 4 -> w[2] = 3^4 - 1 = 80
        # digit 3 = 5 -> w[3] = 3^5 - 1 = 242
        # digit 4 = 3 -> w[4] = 3^3 - 1 = 26
        gen = TokenGenerator(game_type=6453, num_tokens=4, seed=42)

        assert gen.w[1] == 728, f"w[1] should be 728, got {gen.w[1]}"
        assert gen.w[2] == 80, f"w[2] should be 80, got {gen.w[2]}"
        assert gen.w[3] == 242, f"w[3] should be 242, got {gen.w[3]}"
        assert gen.w[4] == 26, f"w[4] should be 26, got {gen.w[4]}"

    def test_token_values_match_java_baseline(self) -> None:
        """
        Validate token values against Java STokenGeneratorOriginal output.

        This test ensures our Python implementation produces IDENTICAL
        token values to the Java reference implementation for the canonical
        game_type=6453, seed=42 configuration.

        Reference values computed from STokenGeneratorOriginal.java.
        """
        # Reference values for game_type=6453, seed=42, num_tokens=4
        # These are the expected outputs from the Java baseline

        # Expected intermediate values (for debugging)
        EXPECTED_A = 65
        EXPECTED_B1 = 62
        EXPECTED_B2 = 53
        EXPECTED_C = [0, 106, 105, 208, 20, 169, 48, 22, 127]

        # Expected token values (the critical reference)
        # These are generated in sequence: new_round(), buyer1, seller1
        EXPECTED_BUYER_1 = [355, 259, 251, 166]
        EXPECTED_SELLER_1 = [143, 179, 267, 308]

        gen = TokenGenerator(game_type=6453, num_tokens=4, seed=42)
        gen.new_round()

        # Validate intermediate values
        assert gen.A == EXPECTED_A, f"A mismatch: {gen.A} != {EXPECTED_A}"
        assert gen.B1 == EXPECTED_B1, f"B1 mismatch: {gen.B1} != {EXPECTED_B1}"
        assert gen.B2 == EXPECTED_B2, f"B2 mismatch: {gen.B2} != {EXPECTED_B2}"
        assert gen.C == EXPECTED_C, f"C mismatch: {gen.C} != {EXPECTED_C}"

        # Generate and validate first buyer's tokens
        buyer_tokens = gen.generate_tokens(is_buyer=True)
        assert buyer_tokens == EXPECTED_BUYER_1, (
            f"Buyer tokens mismatch: {buyer_tokens} != {EXPECTED_BUYER_1}"
        )

        # Generate and validate first seller's tokens
        seller_tokens = gen.generate_tokens(is_buyer=False)
        assert seller_tokens == EXPECTED_SELLER_1, (
            f"Seller tokens mismatch: {seller_tokens} != {EXPECTED_SELLER_1}"
        )


# =============================================================================
# TEST CLASS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for boundary conditions and edge cases."""

    def test_no_bids_high_bidder_zero(self, basic_orderbook: OrderBook) -> None:
        """If no bids submitted, high_bidder should be 0."""
        ob = basic_orderbook
        ob.increment_time()

        # Only submit ask, no bids
        ob.add_ask(1, 50)
        ob.determine_winners()

        assert ob.high_bidder[1] == 0
        assert ob.high_bid[1] == 0

    def test_no_asks_low_asker_zero(self, basic_orderbook: OrderBook) -> None:
        """If no asks submitted, low_asker should be 0."""
        ob = basic_orderbook
        ob.increment_time()

        # Only submit bid, no asks
        ob.add_bid(1, 50)
        ob.determine_winners()

        assert ob.low_asker[1] == 0
        assert ob.low_ask[1] == 0

    def test_trade_at_boundary_prices(self, basic_orderbook: OrderBook) -> None:
        """Trades should work at min_price and max_price boundaries."""
        ob = basic_orderbook
        ob.increment_time()

        # Bid at max, ask at min (extreme case)
        ob.add_bid(1, 100)  # max_price
        ob.add_ask(1, 1)    # min_price
        ob.determine_winners()

        price = ob.execute_trade(buyer_accepts=True, seller_accepts=False)
        assert price == 1  # Trade at ask (min_price)

    def test_single_participant_scenario(self) -> None:
        """Market should handle single buyer/seller gracefully."""
        ob = OrderBook(1, 1, 10, 1, 100, 42)
        ob.increment_time()

        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        assert ob.high_bidder[1] == 1
        assert ob.low_asker[1] == 1

    def test_time_overflow_handled(self) -> None:
        """increment_time should return False when exceeding num_times."""
        ob = OrderBook(2, 2, 3, 1, 100, 42)

        assert ob.increment_time() is True  # t=1
        assert ob.increment_time() is True  # t=2
        assert ob.increment_time() is True  # t=3
        assert ob.increment_time() is False  # t=4, exceeds num_times=3


# =============================================================================
# TEST CLASS: Deadsteps Early Termination
# =============================================================================


class TestDeadstepsTermination:
    """Tests for deadsteps consecutive no-trade early termination feature."""

    def test_deadsteps_disabled_by_default(self) -> None:
        """When deadsteps=0, early termination should be disabled."""
        ob = OrderBook(2, 2, 10, 1, 100, 42, deadsteps=0)

        # Simulate many consecutive no-trades
        for _ in range(20):
            ob.execute_trade(buyer_accepts=False, seller_accepts=False)

        # should_terminate_early should always return False when deadsteps=0
        assert ob.should_terminate_early() is False
        assert ob.consecutive_no_trades == 20

    def test_deadsteps_counter_increments_on_no_trade(self) -> None:
        """Consecutive no-trade counter should increment when no trade occurs."""
        ob = OrderBook(2, 2, 10, 1, 100, 42, deadsteps=5)

        assert ob.consecutive_no_trades == 0

        # First no-trade
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert ob.consecutive_no_trades == 1

        # Second no-trade
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert ob.consecutive_no_trades == 2

        # Third no-trade
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert ob.consecutive_no_trades == 3

    def test_deadsteps_counter_resets_on_trade(self) -> None:
        """Consecutive no-trade counter should reset to 0 when a trade occurs."""
        ob = OrderBook(2, 2, 10, 1, 100, 42, deadsteps=5)

        # Build up consecutive no-trades
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert ob.consecutive_no_trades == 3

        # Set up market for a trade
        ob.increment_time()
        ob.add_bid(1, 50)
        ob.add_ask(1, 60)
        ob.determine_winners()

        # Execute trade (buyer accepts at ask price)
        price = ob.execute_trade(buyer_accepts=True, seller_accepts=False, buyer_id=1, seller_id=1)
        assert price > 0  # Trade occurred
        assert ob.consecutive_no_trades == 0  # Counter reset

    def test_deadsteps_triggers_early_termination(self) -> None:
        """should_terminate_early should return True when threshold reached."""
        ob = OrderBook(2, 2, 10, 1, 100, 42, deadsteps=5)

        # 4 consecutive no-trades - should not trigger
        for _ in range(4):
            ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert ob.should_terminate_early() is False

        # 5th consecutive no-trade - should trigger
        ob.execute_trade(buyer_accepts=False, seller_accepts=False)
        assert ob.consecutive_no_trades == 5
        assert ob.should_terminate_early() is True

    def test_deadsteps_exact_threshold(self) -> None:
        """Termination should trigger at exactly deadsteps consecutive no-trades."""
        for threshold in [1, 3, 5, 10]:
            ob = OrderBook(2, 2, 10, 1, 100, 42, deadsteps=threshold)

            # Execute threshold-1 no-trades
            for _ in range(threshold - 1):
                ob.execute_trade(buyer_accepts=False, seller_accepts=False)
            assert ob.should_terminate_early() is False

            # One more should trigger
            ob.execute_trade(buyer_accepts=False, seller_accepts=False)
            assert ob.consecutive_no_trades == threshold
            assert ob.should_terminate_early() is True
